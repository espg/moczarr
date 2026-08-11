"""Two-store occupancy intersection (cross-sensor cell masks) — issue #39.

The read-side composition primitive englacial/zagg#426 needs (GEDI waveform
stores composed with ATL03 stores): the exact cell occupancy TWO hive stores
share, where the store-vs-AOI helpers (``root_coverage_and`` / ``box_and`` /
``bitmap_and``) answer only one store against a cover. The flow is the §5
tier discipline applied twice:

1. **Root prefilter** — each store's root ``coverage.moc`` (or the discovery
   walk when it is unusable; D9) names its committed shards, and the shared
   regions are their intersection. Shard orders may differ; regions sit at
   the FINER shard order, so each region maps to exactly one leaf per store.
2. **Per-region exact AND** — each leaf's exact occupancy (the ``"bitmap"``
   sidecar, or the whole subtree for ``encoding: "full"``, which
   short-circuits without a sidecar GET) is restricted to the region and
   intersected cell-for-cell.
3. **Order harmonization** — when the stores' cell orders differ (e.g. ATL03
   o19 vs GEDI o18), the finer store's cells coarsen to the coarser order by
   4:1 OR-coarsening (``clip2order`` + unique: a coarse cell is occupied iff
   any of its four children is) before the AND. All results are at
   ``min(cell_order_a, cell_order_b)`` — the manifests set the harmonized
   OUTPUT order, but the order of a leaf's decoded cells is what that
   leaf's OWN envelope declares (``coverage["cell_order"]``, which is what
   the sidecar decoded against). Reading the manifest's value there is a
   silent false negative: a word carries its order in its low bits, so words
   at two orders are never bit-equal and the AND simply matches nothing.

Degradation, documented: a shared leaf WITHOUT exact occupancy at the
harmonized order — a stamp with no usable coverage envelope, a box-only
envelope, a ``"bitmap"`` envelope whose sidecar object is missing, or an
envelope whose declared ``cell_order`` sits BELOW the harmonized order (its
cells name subtrees there, not cells) — contributes its conservative cover
instead (the tier-0 box, the decoded cells read as a cover, or the whole
shard subtree when even the box is unusable). The intersection is then a
conservative SUPERSET for cells under that leaf: false positives possible,
false negatives impossible — the same posture as the box tier everywhere
else — and the first such leaf per call raises a ``UserWarning``. Debris and
absent leaves contribute nothing (absence is definitive; a clean GET miss is
trustworthy on its own).

Both API shapes of the zagg#422 open question 7 are implemented behind one
shared test until the comparison picks the public surface:

- :func:`iter_occupancy_and` — shape (a): an iterator of ``(leaf_id,
  intersected cell words)`` per shared leaf, streaming, matching the
  readers' per-leaf access pattern.
- :func:`occupancy_and` — shape (b): one flat compacted MOC of the whole
  intersection (a fully-shared subtree compacts to its ancestor word, so
  dense overlap stays small).

Spec + fixtures only, per the moczarr contract: everything here decodes
from zagg ``docs/specification.md`` via this package's own convention /
coverage / store layers — no zagg import.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np

from moczarr.convention import morton_word, split_leaf_name
from moczarr.coverage import box_words, parse_leaf_coverage
from moczarr.store import (
    open_object_store,
    read_commits,
    read_coverage_bitmap,
    read_manifest,
)

#: Internal occupancy kinds of one shared leaf (see ``_leaf_occupancy``).
_EMPTY, _FULL, _CELLS, _BOX = "empty", "full", "cells", "box"


class _Occupancy(NamedTuple):
    """One leaf's classified occupancy: kind tag, payload, payload order.

    ``order`` is the order the payload words actually live at — for
    ``"cells"`` the LEAF ENVELOPE's ``cell_order`` (the value the sidecar
    was decoded against), which is the only authority on how far those cells
    must coarsen. ``-1`` where there is no payload.
    """

    kind: str
    words: np.ndarray | None
    order: int = -1


class _Side(NamedTuple):
    """One store's handle plus the intersection's per-store bookkeeping."""

    root: str
    handle: Any
    cell_order: int
    shard_order: int
    leaves: dict[int, str]  # shard word -> store-relative leaf path
    stamps: dict[int, dict | None]  # shard word -> commit stamp (None = debris/absent)


def _open_side(root: str, store: Any, window: str | None, aoi, concurrency, store_kwargs) -> _Side:
    """Open one store and enumerate its candidate leaves (root MOC or walk)."""
    from moczarr.open import _aoi_words, _candidate_leaves

    handle = store if store is not None else open_object_store(root, **store_kwargs)
    manifest = read_manifest(root, store=handle)
    if manifest is None:
        raise ValueError(f"{root!r} has no morton_hive.json — not a hive store root")
    words = None if aoi is None else _aoi_words(aoi)
    leaves: dict[int, str] = {}
    for rel in _candidate_leaves(
        root, manifest, words, window, store=handle, concurrency=concurrency
    ):
        shard, _label = split_leaf_name(rel.rsplit("/", 1)[-1])
        leaves[morton_word(shard)] = rel
    return _Side(
        root, handle, int(manifest["cell_order"]), int(manifest["shard_order"]), leaves, {}
    )


def _shared_regions(a: _Side, b: _Side) -> np.ndarray:
    """Shard-level intersection: ascending region words at the FINER shard order.

    Equal shard orders intersect directly. Otherwise each finer shard is kept
    iff its ancestor at the coarser shard order is a coarser-side shard —
    exact containment, no MOC compaction in the middle (a compacted result
    could sit above BOTH shard orders and stop naming single leaves).
    """
    words_a = np.asarray(sorted(a.leaves), dtype=np.uint64)
    words_b = np.asarray(sorted(b.leaves), dtype=np.uint64)
    if a.shard_order == b.shard_order:
        return np.intersect1d(words_a, words_b)
    fine, coarse = (a, b) if a.shard_order > b.shard_order else (b, a)
    fine_words = words_a if fine is a else words_b
    coarse_words = words_b if fine is a else words_a
    if fine_words.size == 0 or coarse_words.size == 0:
        return np.empty(0, dtype=np.uint64)
    from mortie import clip2order

    ancestors = np.asarray(clip2order(coarse.shard_order, fine_words), dtype=np.uint64)
    return fine_words[np.isin(ancestors, coarse_words)]


def _load_stamps(side: _Side, leaf_words: np.ndarray, concurrency) -> None:
    """Batch the commit-stamp GETs for every participating leaf of one side."""
    unique = sorted({int(w) for w in leaf_words})
    rels = [side.leaves[w] for w in unique]
    stamps = read_commits(side.root, rels, store=side.handle, concurrency=concurrency)
    side.stamps.update(zip(unique, stamps))


def _leaf_occupancy(side: _Side, word: int, warned: list, out_order: int) -> _Occupancy:
    """Classify one leaf's occupancy: kind tag, payload, payload order.

    ``"empty"`` for debris/absent (nothing there — definitive); ``"full"``
    for ``encoding: "full"`` (whole subtree, exact, no sidecar GET — the
    short-circuit) AND for the conservative stand-in when a stamped leaf has
    no usable envelope at all; ``"cells"`` for a decoded bitmap sidecar
    whose envelope order can still be made exact at ``out_order``;
    ``"box"`` for a conservative cover — the tier-0 box (box-only envelopes,
    and bitmap envelopes whose sidecar object is missing), or decoded cells
    read AS a cover when the envelope's ``cell_order`` is below
    ``out_order`` (those words name subtrees at the harmonized order, and
    refining them would invent occupancy). The conservative kinds warn once
    per call.
    """
    rel = side.leaves[word]
    stamp = side.stamps.get(word)
    if stamp is None:
        return _Occupancy(_EMPTY, None)
    coverage = parse_leaf_coverage(stamp)
    if coverage is None:
        _warn_degraded(warned, side, rel, "stamp carries no usable coverage envelope")
        return _Occupancy(_FULL, None)
    if coverage.get("encoding") == "full":
        return _Occupancy(_FULL, None)
    if coverage.get("encoding") == "bitmap":
        cells = read_coverage_bitmap(side.root, rel, coverage=coverage, store=side.handle)
        if cells is not None:
            env_order = int(coverage["cell_order"])
            if env_order >= out_order:
                return _Occupancy(_CELLS, cells, env_order)
            _warn_degraded(
                warned,
                side,
                rel,
                f"envelope cell_order {env_order} is below the harmonized order {out_order}",
            )
            return _Occupancy(_BOX, cells, env_order)
    try:
        box = np.asarray(box_words(coverage), dtype=np.uint64)
    except (KeyError, TypeError, ValueError):
        box = np.empty(0, dtype=np.uint64)
    if box.size == 0:
        _warn_degraded(warned, side, rel, "coverage envelope has no exact encoding and no box")
        return _Occupancy(_FULL, None)
    _warn_degraded(warned, side, rel, "no exact occupancy (box-only envelope or missing sidecar)")
    return _Occupancy(_BOX, box)


def _warn_degraded(warned: list, side: _Side, rel: str, why: str) -> None:
    """One ``UserWarning`` per call for the first leaf without exact occupancy."""
    if warned:
        return
    warned.append(rel)
    warnings.warn(
        f"leaf {rel} of {side.root} contributes a conservative cover ({why}): the "
        f"intersection is a SUPERSET for cells under such leaves — false positives "
        f"possible, false negatives impossible. Further degraded leaves in this call "
        f"do not re-warn.",
        stacklevel=2,
    )


def _expand_to(words: np.ndarray, order: int) -> np.ndarray:
    """A mixed-order cover as unique ascending cells at ``order``.

    Members above ``order`` refine to their descendants (``children_of``,
    batched per distinct order); members at-or-below clip. The exact inverse
    of 4:1 OR-coarsening for full subtrees.
    """
    if words.size == 0:
        return np.empty(0, dtype=np.uint64)
    from mortie import children_of, clip2order, orders_of

    orders = np.asarray(orders_of(words))
    parts = []
    deeper = words[orders >= order]
    if deeper.size:
        parts.append(np.unique(np.asarray(clip2order(order, deeper), dtype=np.uint64)))
    shallow = words[orders < order]
    for o in np.unique(orders[orders < order]):
        group = shallow[np.asarray(orders_of(shallow)) == o]
        parts.append(np.asarray(children_of(group, order), dtype=np.uint64).ravel())
    return np.unique(np.concatenate(parts))


def _region_cells(
    side: _Side, occ: _Occupancy, region: int, region_order: int, out_order: int
) -> np.ndarray | None:
    """One side's occupancy inside ``region``, harmonized to ``out_order``.

    ``None`` means the whole region subtree (full — exact or conservative);
    otherwise sorted unique cell words at ``out_order``. Exact bitmap cells
    restrict to the region by ancestor equality (their envelope order is
    at-or-below ``out_order``, which the compose guard puts at-or-below
    ``region_order``, so the clip is always legal), then 4:1 OR-coarsen when
    the ENVELOPE's order out-resolves ``out_order``. A conservative cover
    intersects the region as a MOC and expands.
    """
    from mortie import clip2order

    if occ.kind == _EMPTY:
        return np.empty(0, dtype=np.uint64)
    if occ.kind == _FULL:
        return None
    if occ.kind == _CELLS:
        cells = occ.words
        if side.shard_order < region_order:
            ancestors = np.asarray(clip2order(region_order, cells), dtype=np.uint64)
            cells = cells[ancestors == np.uint64(region)]
        if occ.order > out_order:
            cells = np.unique(np.asarray(clip2order(out_order, cells), dtype=np.uint64))
        return cells
    from mortie import moc_and

    overlap = np.asarray(moc_and(occ.words, np.asarray([region], dtype=np.uint64)), dtype=np.uint64)
    return _expand_to(overlap, out_order)


def _iter_shared(
    root_a: str,
    root_b: str,
    aoi,
    window_a: str | None,
    window_b: str | None,
    store_a: Any,
    store_b: Any,
    concurrency,
    store_kwargs: dict[str, Any],
    *,
    expand_full: bool,
) -> Iterator[tuple[int, np.ndarray | None]]:
    """The shared core both API shapes consume.

    Yields ``(region word, cells)`` per shared region with a nonempty
    intersection. A region full on BOTH sides yields its whole subtree —
    expanded to cells when ``expand_full`` (shape a), or ``None`` so shape
    (b) can keep the region word compact.
    """
    a = _open_side(root_a, store_a, window_a, aoi, concurrency, store_kwargs)
    b = _open_side(root_b, store_b, window_b, aoi, concurrency, store_kwargs)
    out_order = min(a.cell_order, b.cell_order)
    region_order = max(a.shard_order, b.shard_order)
    if out_order < region_order:
        raise ValueError(
            f"stores do not compose: cell order {out_order} of the coarser store is "
            f"above shard order {region_order} of the finer-sharded store, so a shared "
            f"region has no cell representation in both stores"
        )
    regions = _shared_regions(a, b)
    if regions.size == 0:
        return
    from mortie import clip2order

    leaf_words = {}
    for side in (a, b):
        if side.shard_order == region_order:
            leaf_words[id(side)] = regions
        else:
            leaf_words[id(side)] = np.asarray(
                clip2order(side.shard_order, regions), dtype=np.uint64
            )
        _load_stamps(side, leaf_words[id(side)], concurrency)
    warned: list = []
    occupancy: dict[tuple[int, int], _Occupancy] = {}
    for i, region in enumerate(int(r) for r in regions):
        cells = []
        for side in (a, b):
            leaf = int(leaf_words[id(side)][i])
            key = (id(side), leaf)
            if key not in occupancy:
                occupancy[key] = _leaf_occupancy(side, leaf, warned, out_order)
            cells.append(_region_cells(side, occupancy[key], region, region_order, out_order))
        if cells[0] is None and cells[1] is None:
            full = np.asarray([region], dtype=np.uint64)
            yield region, _expand_to(full, out_order) if expand_full else None
            continue
        if cells[0] is None or cells[1] is None:
            hit = cells[1] if cells[0] is None else cells[0]
        else:
            hit = np.intersect1d(cells[0], cells[1])
        if hit.size:
            yield region, hit


def iter_occupancy_and(
    root_a: str,
    root_b: str,
    *,
    aoi=None,
    window_a: str | None = None,
    window_b: str | None = None,
    store_a: Any = None,
    store_b: Any = None,
    concurrency: int | None = 32,
    **store_kwargs: Any,
) -> Iterator[tuple[int, np.ndarray]]:
    """Exact shared occupancy of two hive stores, streamed per shared leaf.

    Shape (a) of the zagg#422 open question 7: yields ``(leaf_id, cells)``
    in ascending packed-word order — ``leaf_id`` the shared region's packed
    shard word (the finer store's leaf when shard orders differ; both
    stores' when equal) and ``cells`` the sorted occupied cell words BOTH
    stores share under it, at the coarser of the two cell orders (4:1
    OR-coarsening harmonizes a finer store — a coarse cell is occupied iff
    any of its four children is). Regions whose exact intersection is empty
    are skipped; an exhausted iterator with no yields means the stores share
    no occupancy.

    ``aoi`` (a morton cover: packed words or decimal strings, mixed orders
    allowed) prefilters SHARDS — leaves whose subtree misses the cover never
    load — but cells of a kept leaf are not themselves clipped; apply
    :func:`moczarr.coverage.aoi_mask` to the yielded cells for an exact
    cell-level cut. ``window_a`` / ``window_b`` name the time window per
    store (``morton-hive/2``); ``store_a`` / ``store_b`` pass pre-built
    obstore handles (required when the two stores need different
    credentials — ``store_kwargs`` apply to BOTH roots otherwise).
    ``concurrency`` bounds the batched stamp GETs and walk LISTs.

    Degradation and its ``UserWarning`` are per the module docstring: leaves
    without exact occupancy contribute their conservative cover, making the
    result a superset under those leaves only. Raises ``ValueError`` when a
    root has no manifest, or when the orders do not compose (one store's
    cell order above the other's shard order).
    """
    yield from _iter_shared(
        root_a,
        root_b,
        aoi,
        window_a,
        window_b,
        store_a,
        store_b,
        concurrency,
        store_kwargs,
        expand_full=True,
    )


def occupancy_and(
    root_a: str,
    root_b: str,
    *,
    aoi=None,
    window_a: str | None = None,
    window_b: str | None = None,
    store_a: Any = None,
    store_b: Any = None,
    concurrency: int | None = 32,
    **store_kwargs: Any,
) -> np.ndarray:
    """Exact shared occupancy of two hive stores as one flat compacted MOC.

    Shape (b) of the zagg#422 open question 7: the same intersection
    :func:`iter_occupancy_and` streams, returned whole as a canonical
    compact morton cover (``uint64``, ascending) — a fully-shared subtree
    compacts to its ancestor word, so dense overlap stays small, and the
    empty intersection is an empty array. Expanding every member to the
    coarser cell order yields exactly the union of the iterator's cells
    (pinned by the shared golden test). Parameters, degradation, and errors
    are identical to :func:`iter_occupancy_and`.
    """
    parts = [
        np.asarray([region], dtype=np.uint64) if cells is None else cells
        for region, cells in _iter_shared(
            root_a,
            root_b,
            aoi,
            window_a,
            window_b,
            store_a,
            store_b,
            concurrency,
            store_kwargs,
            expand_full=False,
        )
    ]
    if not parts:
        return np.empty(0, dtype=np.uint64)
    from mortie import compress_moc

    return np.sort(np.asarray(compress_moc(np.concatenate(parts)), dtype=np.uint64))


__all__ = ["iter_occupancy_and", "occupancy_and"]

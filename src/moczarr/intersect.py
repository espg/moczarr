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
else — and the first such leaf per call raises a
:class:`moczarr.exceptions.ConservativeCoverageWarning` (a ``UserWarning``
subclass, so it stays filterable by class without catching everything). That is
the ``degrade="conservative"`` default; ``degrade="skip"`` drops such leaves
instead (nothing is yielded under them, so everything returned IS exact —
exact-or-under rather than superset) and ``degrade="raise"`` refuses the
call, naming the first one. Debris and absent leaves contribute nothing
under every policy (absence is definitive; a clean GET miss is trustworthy
on its own).

Two API shapes, ruled on the measured comparison of zagg#422 open
question 7 — one is the surface, the other is derived from it:

- :func:`iter_occupancy_and` — shape (a), THE public surface: an iterator
  of ``(leaf_id, intersected cell words)`` per shared leaf, streaming, and
  the shape the zagg#426 consumer reads through (leaf identity is what
  fetches the payloads for the aligned per-leaf tensors).
- :func:`occupancy_and` — shape (b), a documented thin convenience DERIVED
  from (a): one flat compacted MOC of the whole intersection, semantically
  ``compress_moc`` over (a)'s stream (a fully-shared subtree compacts to
  its ancestor word, so dense overlap stays small). It is the currency for
  MOC algebra — it feeds ``open_hive(aoi=...)``, ``moc_and`` and
  :func:`moczarr.coverage.aoi_mask` directly, and is the compact thing to
  persist or ship.

The derivation runs ONE way: (b) is one ``compress_moc`` over (a), while
(a) is NOT recoverable from (b) — compaction discards the leaf attribution
that per-leaf reads need. Reach for (a) to read cells under a leaf, (b) to
do MOC algebra with the answer.

``aoi=`` is SHARD-level on both, by contract: it restricts the shard cover
through the same candidate-leaves arithmetic the opener uses, and the cells
of a kept leaf are never clipped to it. Both shapes are therefore supersets
with respect to the AOI — false positives possible, false negatives
impossible, zagg's AOI-overhang convention — and
:func:`moczarr.coverage.aoi_mask` is the documented exact cell-level cut.
Keeping the clip out of here is what lets a fully-shared subtree stay one
compact word under a partial AOI instead of expanding to be trimmed.

The AND itself runs in MOC currency and never materializes a subtree:
exact cells against a conservative cover are filtered by containment
(``coverage.aoi_mask``), cover against cover is a ``moc_and``, so the work
is bounded by the exact side wherever one exists. Cells are materialized
only where shape (a) has to yield them, and only for a region that is a
cover on BOTH sides — full-on-both (exact by construction; its whole
subtree IS the answer) or degraded on both — at
``4^(out_order - region_order)`` words. Shape (b) never expands. That
asymmetry is itself an input to the zagg#422 shape decision.

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
from moczarr.coverage import aoi_mask, box_words, parse_leaf_coverage
from moczarr.exceptions import ConservativeCoverageWarning
from moczarr.store import (
    open_object_store,
    read_commits,
    read_coverage_bitmap,
    read_manifest,
)

#: Internal occupancy kinds of one shared leaf (see ``_leaf_occupancy``).
_EMPTY, _FULL, _CELLS, _BOX = "empty", "full", "cells", "box"
#: Internal currencies of one region's contribution (see ``_region_part``).
_EXACT, _COVER = "exact", "cover"
#: ``degrade=`` policies for a leaf without exact occupancy (see ``_degrade``).
_CONSERVATIVE, _SKIP, _RAISE = "conservative", "skip", "raise"
_DEGRADE_MODES = (_CONSERVATIVE, _SKIP, _RAISE)


class _Part(NamedTuple):
    """One side's contribution to ONE region, in one of two currencies.

    ``"exact"`` — cell words AT ``order`` (the harmonized order), the truth.
    ``"cover"`` — a compact mixed-order MOC (members between the region's
    own order and ``order``) standing in conservatively for cells that are
    not known exactly. A cover is NEVER expanded to cells inside the
    intersection: a degraded leaf at zagg's depths is 4^13 words, and every
    AND below has a form that keeps it compact. ``order`` travels with the
    part so a consumer that does want cells knows what to expand to.
    """

    kind: str
    words: np.ndarray
    order: int


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


class _Plan(NamedTuple):
    """One call's prepared state: everything :func:`_setup` resolves eagerly.

    ``regions`` are the shared shard words at ``region_order``; ``leaf_words``
    maps ``id(side)`` to that side's participating leaf per region (index
    aligned with ``regions``), with its commit stamps already fetched.
    ``out_order`` is the harmonized cell order every result sits at, and
    ``degrade`` the policy for leaves that cannot answer exactly there.
    """

    a: _Side
    b: _Side
    regions: np.ndarray
    leaf_words: dict[int, np.ndarray]
    region_order: int
    out_order: int
    degrade: str


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


def _leaf_occupancy(
    side: _Side, word: int, warned: list, out_order: int, policy: str
) -> _Occupancy:
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
    refining them would invent occupancy). Every conservative outcome goes
    through :func:`_degrade`, which the caller's ``degrade=`` policy governs.
    """
    rel = side.leaves[word]
    stamp = side.stamps.get(word)
    if stamp is None:
        return _Occupancy(_EMPTY, None)
    coverage = parse_leaf_coverage(stamp)
    if coverage is None:
        why = "stamp carries no usable coverage envelope"
        return _degrade(policy, warned, side, rel, why, _Occupancy(_FULL, None))
    if coverage.get("encoding") == "full":
        return _Occupancy(_FULL, None)
    if coverage.get("encoding") == "bitmap":
        cells = read_coverage_bitmap(side.root, rel, coverage=coverage, store=side.handle)
        if cells is not None:
            env_order = int(coverage["cell_order"])
            if env_order >= out_order:
                return _Occupancy(_CELLS, cells, env_order)
            why = f"envelope cell_order {env_order} is below the harmonized order {out_order}"
            return _degrade(policy, warned, side, rel, why, _Occupancy(_BOX, cells, env_order))
    try:
        box = np.asarray(box_words(coverage), dtype=np.uint64)
    except (KeyError, TypeError, ValueError):
        box = np.empty(0, dtype=np.uint64)
    if box.size == 0:
        why = "coverage envelope has no exact encoding and no box"
        return _degrade(policy, warned, side, rel, why, _Occupancy(_FULL, None))
    why = "box-only envelope or missing sidecar"
    return _degrade(policy, warned, side, rel, why, _Occupancy(_BOX, box))


def _degrade(
    policy: str, warned: list, side: _Side, rel: str, why: str, fallback: _Occupancy
) -> _Occupancy:
    """Apply the call's ``degrade=`` policy to one leaf without exact occupancy.

    ``"conservative"`` (the default) takes ``fallback`` — the leaf's
    conservative cover — and raises ONE
    :class:`moczarr.exceptions.ConservativeCoverageWarning` per call, for
    the first such leaf. ``"skip"`` drops the leaf: it contributes nothing, so
    the call's result is exact-or-under rather than a superset, which is the
    discriminator a consumer keying on per-cell truth needs. ``"raise"``
    stops and names the first such leaf.

    No ``stacklevel`` on the warning: the leaf is classified inside a
    generator driven by the consumer, so no stack level reaches the call
    site — the message names the store and leaf instead, and the dedicated
    category is what a filter should key on (module-scoped ``-W`` filters
    on ``moczarr.intersect`` work too).
    """
    if policy == _RAISE:
        raise ValueError(
            f"leaf {rel} of {side.root} has no exact occupancy ({why}) and degrade='raise' "
            f"refuses a conservative answer — use degrade='conservative' to accept the "
            f"superset, or degrade='skip' to drop such leaves"
        )
    if policy == _SKIP:
        return _Occupancy(_EMPTY, None)
    if not warned:
        warned.append(rel)
        warnings.warn(
            f"leaf {rel} of {side.root} contributes a conservative cover ({why}): the "
            f"intersection is a SUPERSET for cells under such leaves — false positives "
            f"possible, false negatives impossible. Further degraded leaves in this call "
            f"do not re-warn; degrade='skip' drops them and degrade='raise' refuses them.",
            ConservativeCoverageWarning,
        )
    return fallback


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


def _clip_cover(words: np.ndarray, order: int) -> np.ndarray:
    """A cover with no member finer than ``order`` — deeper members clip up.

    Enlarging a member to its ancestor is conservative in the direction the
    cover tier already allows (superset), and it caps every cover in this
    module at ``out_order`` so a cover can never out-resolve the exact tier
    it is ANDed against. ``clip2order`` is per-element: coarser members pass
    through untouched.
    """
    if words.size == 0:
        return np.empty(0, dtype=np.uint64)
    from mortie import clip2order

    return np.unique(np.asarray(clip2order(order, words), dtype=np.uint64))


def _region_part(
    side: _Side, occ: _Occupancy, region: int, region_order: int, out_order: int
) -> _Part:
    """One side's contribution to ``region``: exact cells, or a compact cover.

    Exact bitmap cells restrict to the region by ancestor equality (their
    envelope order is at-or-above ``out_order``, which the compose guard puts
    at-or-above ``region_order``, so the clip is always legal), then 4:1
    OR-coarsen when the ENVELOPE's order out-resolves ``out_order`` — an
    ``"exact"`` part, always at ``out_order``.

    Everything conservative stays a ``"cover"``: mixed-order MOC members
    intersected with the region (``moc_and``, so members land at-or-below
    ``region_order``) and clipped up to ``out_order``. ``"full"`` is just
    the cover ``[region]``. Nothing here materializes a subtree — the whole
    point of keeping covers compact (a degraded leaf at ATL03 depths is
    4^13 cells).
    """
    from mortie import clip2order

    if occ.kind == _EMPTY:
        return _Part(_EXACT, np.empty(0, dtype=np.uint64), out_order)
    if occ.kind == _FULL:
        return _Part(_COVER, np.asarray([region], dtype=np.uint64), out_order)
    if occ.kind == _CELLS:
        cells = occ.words
        if side.shard_order < region_order:
            ancestors = np.asarray(clip2order(region_order, cells), dtype=np.uint64)
            cells = cells[ancestors == np.uint64(region)]
        if occ.order > out_order:
            cells = np.unique(np.asarray(clip2order(out_order, cells), dtype=np.uint64))
        return _Part(_EXACT, cells, out_order)
    from mortie import moc_and

    overlap = np.asarray(moc_and(occ.words, np.asarray([region], dtype=np.uint64)), dtype=np.uint64)
    return _Part(_COVER, _clip_cover(overlap, out_order), out_order)


def _and_parts(part_a: _Part, part_b: _Part, out_order: int) -> _Part:
    """AND two region parts, staying in the cheaper of the two currencies.

    - exact ∧ exact — ``np.intersect1d``; both already sit at ``out_order``.
    - exact ∧ cover — keep the exact side's cells that meet the cover
      (:func:`moczarr.coverage.aoi_mask`, the house two-way containment
      predicate — NOT ``isin`` against a compacted ``moc_and``, which drops
      exactly the dense subtrees). Conservative-correct, and the result is
      bounded by the exact side, so a degraded leaf costs nothing extra.
    - cover ∧ cover — ``moc_and`` of the two covers, still a cover.

    An expansion to cells happens NOWHERE here; shape (a) does it once, at
    the end, and only for a region that stayed a cover on both sides.
    """
    if part_a.kind == _EXACT and part_b.kind == _EXACT:
        return _Part(_EXACT, np.intersect1d(part_a.words, part_b.words), out_order)
    if part_a.kind == _EXACT or part_b.kind == _EXACT:
        exact, cover = (part_a, part_b) if part_a.kind == _EXACT else (part_b, part_a)
        if exact.words.size == 0 or cover.words.size == 0:
            return _Part(_EXACT, np.empty(0, dtype=np.uint64), out_order)
        return _Part(_EXACT, exact.words[aoi_mask(exact.words, cover.words)], out_order)
    if part_a.words.size == 0 or part_b.words.size == 0:
        return _Part(_COVER, np.empty(0, dtype=np.uint64), out_order)
    from mortie import moc_and

    members = np.asarray(moc_and(part_a.words, part_b.words), dtype=np.uint64)
    return _Part(_COVER, _clip_cover(members, out_order), out_order)


def _setup(
    root_a: str,
    root_b: str,
    aoi,
    window_a: str | None,
    window_b: str | None,
    store_a: Any,
    store_b: Any,
    concurrency,
    store_kwargs: dict[str, Any],
    degrade: str,
) -> _Plan:
    """Everything both shapes do EAGERLY, before the first region is asked for.

    Validates ``degrade``, opens both stores, validates the orders, resolves
    the shared regions and each side's participating leaf per region, and
    batches the commit-stamp GETs. Split out so shape (a) can be a plain
    function returning a generator: a bad root, a bad knob, or an
    incomposable pair of stores is a call-time ``ValueError``, not a surprise
    at the consumer's first ``next()`` in some unrelated frame.
    """
    if degrade not in _DEGRADE_MODES:
        raise ValueError(f"degrade={degrade!r} is not one of {_DEGRADE_MODES}")
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
    leaf_words: dict[int, np.ndarray] = {}
    if regions.size:
        from mortie import clip2order

        for side in (a, b):
            if side.shard_order == region_order:
                leaf_words[id(side)] = regions
            else:
                leaf_words[id(side)] = np.asarray(
                    clip2order(side.shard_order, regions), dtype=np.uint64
                )
            _load_stamps(side, leaf_words[id(side)], concurrency)
    return _Plan(a, b, regions, leaf_words, region_order, out_order, degrade)


def _iter_parts(plan: _Plan) -> Iterator[tuple[int, _Part]]:
    """The shared core both API shapes consume, over a prepared :class:`_Plan`.

    Yields ``(region word, part)`` per shared region with a nonempty
    intersection — an ``"exact"`` part whenever EITHER side knew its cells
    exactly, otherwise the compact ``"cover"`` both sides agreed on (a
    region full on both, or degraded on both). Shape (a) expands that cover;
    shape (b) carries it straight into the flat MOC.
    """
    warned: list = []
    occupancy: dict[tuple[int, int], _Occupancy] = {}
    for i, region in enumerate(int(r) for r in plan.regions):
        parts = []
        for side in (plan.a, plan.b):
            leaf = int(plan.leaf_words[id(side)][i])
            key = (id(side), leaf)
            if key not in occupancy:
                occupancy[key] = _leaf_occupancy(side, leaf, warned, plan.out_order, plan.degrade)
            parts.append(
                _region_part(side, occupancy[key], region, plan.region_order, plan.out_order)
            )
        hit = _and_parts(parts[0], parts[1], plan.out_order)
        if hit.words.size:
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
    degrade: str = _CONSERVATIVE,
    **store_kwargs: Any,
) -> Iterator[tuple[int, np.ndarray]]:
    """Exact shared occupancy of two hive stores, streamed per shared leaf.

    Shape (a), and the module's public surface (zagg#422 open question 7,
    ruled on this PR's measured comparison): the shape per-leaf readers
    want, because ``leaf_id`` is what fetches a leaf's payloads — the
    zagg#426 flow builds its aligned side-by-side tensors that way
    (``moczarr.hhdc.read_tensors`` / :func:`moczarr.open_leaf` per yielded
    leaf). :func:`occupancy_and` is one ``compress_moc`` over this stream;
    the reverse does not exist, since compaction discards leaf attribution.

    Yields ``(leaf_id, cells)``
    in ascending packed-word order — ``leaf_id`` the shared region's packed
    shard word (the finer store's leaf when shard orders differ; both
    stores' when equal) and ``cells`` the sorted occupied cell words BOTH
    stores share under it, at the coarser of the two cell orders (4:1
    OR-coarsening harmonizes a finer store — a coarse cell is occupied iff
    any of its four children is). Regions whose exact intersection is empty
    are skipped; an exhausted iterator with no yields means the stores share
    no occupancy.

    ``aoi`` (a morton cover: packed words or decimal strings, mixed orders
    allowed) is a SHARD-LEVEL prefilter, and that is the contract, not an
    implementation detail: it runs through the same candidate-leaves
    arithmetic :func:`moczarr.open_hive` uses, so a leaf whose subtree
    misses the cover never loads — but the cells of a KEPT leaf are NOT
    clipped to the cover. Results are therefore a SUPERSET with respect to
    the AOI: cells outside it can be yielded (false positives possible),
    cells inside it are never dropped (false negatives impossible). That is
    zagg's AOI-overhang convention, the same one-directional posture the
    box tier keeps everywhere else here, and it is what lets a region full
    on both sides stay compact under a partial AOI instead of expanding to
    be clipped. The documented EXACT cell-level cut is
    :func:`moczarr.coverage.aoi_mask` over the yielded cells:
    ``cells[aoi_mask(cells, aoi_words)]``.

    ``window_a`` / ``window_b`` name the time window per
    store (``morton-hive/2``); ``store_a`` / ``store_b`` pass pre-built
    obstore handles (required when the two stores need different
    credentials — ``store_kwargs`` apply to BOTH roots otherwise).
    ``concurrency`` bounds the batched stamp GETs and walk LISTs.

    ``degrade`` picks what a leaf without exact occupancy at the harmonized
    order does — the module docstring lists which leaves those are:
    ``"conservative"`` (default) contributes its cover, so the result is a
    SUPERSET under that leaf (false positives possible, false negatives
    impossible) and the first such leaf per call raises one
    :class:`moczarr.exceptions.ConservativeCoverageWarning` — its own
    ``UserWarning`` subclass, so a consumer can promote it to an error or
    silence it by category;
    ``"skip"`` drops it, so the leaf contributes nothing and EVERYTHING
    yielded is exact — exact-or-under instead of superset, the discriminator
    a per-cell consumer needs — at the price of silently missing real shared
    occupancy under it; ``"raise"`` refuses, naming the first such leaf.
    Raises ``ValueError`` AT CALL
    TIME — not at the first ``next()`` — when a root has no manifest, or when
    the orders do not compose (one store's cell order above the other's shard
    order): both stores are opened and validated before the iterator exists,
    so a store-root typo surfaces where it was made.

    Cost: yielding CELLS is what shape (a) is for, and a region whose result
    is a cover on BOTH sides — full-on-both (exact, and bounded by the
    region's own subtree), or degraded on both — materializes
    ``4^(out_order - region_order)`` words for that one yield. Everywhere
    else the intersection is bounded by the exact side and nothing is
    expanded, degraded leaves included. :func:`occupancy_and` never expands
    at all; reach for it when the harmonized order is far below the shard
    order.
    """
    return _iter_cells(
        _setup(
            root_a,
            root_b,
            aoi,
            window_a,
            window_b,
            store_a,
            store_b,
            concurrency,
            store_kwargs,
            degrade,
        )
    )


def _iter_cells(plan: _Plan) -> Iterator[tuple[int, np.ndarray]]:
    """Shape (a)'s generator half: parts as cells, covers expanded once."""
    for region, part in _iter_parts(plan):
        cells = part.words if part.kind == _EXACT else _expand_to(part.words, part.order)
        if cells.size:
            yield region, cells


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
    degrade: str = _CONSERVATIVE,
    **store_kwargs: Any,
) -> np.ndarray:
    """The :func:`iter_occupancy_and` intersection, compacted to one flat MOC.

    Shape (b): a thin convenience DERIVED from the public surface —
    semantically ``compress_moc`` over everything
    :func:`iter_occupancy_and` yields, returned whole as a canonical
    compact morton cover (``uint64``, ascending). A fully-shared subtree
    compacts to its ancestor word, so dense overlap stays small, and the
    empty intersection is an empty array. Parameters (``degrade``
    included), degradation, and errors are identical to
    :func:`iter_occupancy_and`.

    ``aoi`` keeps the same contract it has there — a SHARD-LEVEL prefilter,
    with the cells of a kept leaf left unclipped, so the returned MOC is a
    superset with respect to the AOI (false positives possible, false
    negatives impossible: zagg's AOI-overhang convention). Worth restating
    on this shape, because a MOC is what gets composed onward: feed the
    result to ``moc_and(result, aoi_words)`` (or
    :func:`moczarr.coverage.aoi_mask` on cells) when the consumer needs the
    exact cut rather than the overhang.

    The derivation is one-way, and that asymmetry is what picks between
    the two:

    - **(b) from (a)** is one line —
      ``compress_moc(concatenate([cells for _leaf, cells in
      iter_occupancy_and(...)]))`` — and equals this function's result
      member-for-member (pinned by ``test_flat_moc_is_the_compressed_stream``).
    - **(a) from (b)** does not exist: compaction discards leaf
      attribution, and a member may sit above, at, or below a leaf. Use
      shape (a) for per-leaf reads — the zagg#426 tensor path, which needs
      the leaf id to fetch payloads — and this shape as MOC-algebra
      currency: it is what ``open_hive(aoi=...)``, ``moc_and`` and
      :func:`moczarr.coverage.aoi_mask` consume, and the compact thing to
      persist or ship.

    Cost — the reason this is NOT implemented as the literal one-liner:
    the composition above would expand a region that is full (or degraded)
    on both sides to its ``4^(out_order - region_order)`` cells, purely to
    compact them back to the ancestor word they came from (~262k words at
    the zagg#426 o9-shard/o18-cell scale, per region). This stays in MOC
    currency end to end instead — same result, no subtree ever
    materialized (pinned by ``test_only_the_iterator_expands_full_on_both``).
    Expanding every member back to the harmonized cell order yields exactly
    the union of the iterator's cells (the shared golden test).
    """
    plan = _setup(
        root_a,
        root_b,
        aoi,
        window_a,
        window_b,
        store_a,
        store_b,
        concurrency,
        store_kwargs,
        degrade,
    )
    parts = [part.words for _region, part in _iter_parts(plan)]
    if not parts:
        return np.empty(0, dtype=np.uint64)
    from mortie import compress_moc

    return np.sort(np.asarray(compress_moc(np.concatenate(parts)), dtype=np.uint64))


__all__ = ["iter_occupancy_and", "occupancy_and"]

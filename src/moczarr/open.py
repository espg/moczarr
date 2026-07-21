"""``open_hive``: a morton-hive store as one lazy xarray Dataset.

The §5 reader flow, arithmetic-first (D10): manifest GET → root coverage
MOC ∩ AOI → hive paths by string arithmetic → stamped-leaf opens → concat
along the cell dimension. The discovery walk runs only when the root MOC is
absent/unusable (D9: caches degrade to the walk, never to wrong answers).
Debris (unstamped leaves) is skipped silently — absence of a stamp IS the
answer (D4).

AOI semantics: ``aoi`` is a morton cover — packed ``uint64`` words or
decimal strings, mixed orders allowed. Shards are rejected arithmetically
(root MOC ∩ AOI), then per leaf by the tier-0 box off the stamp already
fetched, and finally rows are subset exactly on the ``morton`` coordinate
(tier 2 — the coordinate is the truth; the MOC tiers are only indexes).

An AOI (or window) that intersects no coverage is a data answer, not an
error: the result is a schema-correct EMPTY dataset plus a ``UserWarning``
(issue #4). Only a store with no stamped coverage anywhere — no schema
source at all — raises, as :class:`moczarr.exceptions.NoCoverageError`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from moczarr.convention import (
    HIVE_SPEC_V2,
    leaf_path,
    manifest_path_grouping,
    morton_word,
    split_leaf_name,
    validate_label,
)
from moczarr.coverage import (
    aoi_mask,
    box_and,
    parse_leaf_coverage,
    ranges_words,
    root_coverage_and,
)
from moczarr.exceptions import NoCoverageError
from moczarr.fabricate import fabricate_cell_ids as _fabricate_cell_ids
from moczarr.store import (
    load_root_coverage,
    open_object_store,
    read_commits,
    read_manifest,
    walk_leaves,
)


def _aoi_words(aoi) -> np.ndarray:
    """Normalize an AOI cover to packed ``uint64`` words (strings accepted)."""
    values = list(np.asarray(aoi).ravel()) if np.asarray(aoi).ndim else [aoi]
    return np.asarray([morton_word(v) for v in values], dtype=np.uint64)


def _candidate_leaves(
    store_root: str,
    manifest: dict,
    aoi,
    window: str | None,
    *,
    store: Any = None,
    concurrency: int | None = None,
) -> list[str]:
    """Store-relative leaf paths to try, ascending in packed-word order.

    Arithmetic (root MOC) when possible; the walk otherwise. A windowed
    (``morton-hive/2``) store needs an explicit ``window`` for the
    arithmetic path — with ``window=None`` the walk enumerates what exists
    and the error message lists the labels.
    """
    windowed = manifest["spec"] == HIVE_SPEC_V2
    grouping = manifest_path_grouping(manifest)
    if window is not None:
        validate_label(window)
        if not windowed:
            raise ValueError(
                f"window={window!r} on a {manifest['spec']} store: unwindowed stores "
                f"have no window leaves (schedule: none)"
            )
    envelope = load_root_coverage(store_root, store=store)
    if envelope is not None and not (windowed and window is None):
        words = ranges_words(envelope) if aoi is None else root_coverage_and(envelope, aoi)
        if aoi is not None and words.size:
            # The MOC intersection keeps the FINER element of each overlapping
            # pair (a cell-order AOI member intersected with its shard yields
            # the cell), so coarsen back to shard order to name the leaves.
            from mortie import clip2order

            words = np.unique(clip2order(int(manifest["shard_order"]), words))
        return [leaf_path(int(w), window=window, path_grouping=grouping) for w in np.sort(words)]
    # Walk fallback (no usable root MOC), and the windowed-discovery case.
    found: dict[str, int] = {}
    labels: set[str] = set()
    for rel in walk_leaves(
        store_root, store=store, concurrency=concurrency, path_grouping=grouping
    ):
        shard, label = split_leaf_name(rel.rsplit("/", 1)[-1])
        labels.add(label if label is not None else "<none>")
        if label != window:
            continue
        found[rel] = morton_word(shard)
    if windowed and window is None and labels - {"<none>"}:
        raise ValueError(
            f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; pass window=... "
            f"(labels present: {sorted(labels - {'<none>'})})"
        )
    if aoi is not None:
        from mortie import moc_and

        found = {
            rel: w
            for rel, w in found.items()
            if moc_and(np.asarray([w], dtype=np.uint64), aoi).size
        }
    return sorted(found, key=lambda rel: found[rel])


def _schema_leaf(
    store_root: str,
    window: str | None,
    *,
    store: Any = None,
    concurrency: int | None = None,
    path_grouping: int = 1,
) -> str | None:
    """One commit-stamped leaf anywhere in the store, or ``None`` (issue #4).

    The empty-AOI return needs a schema source — data-variable names,
    dtypes, and attrs live only in leaf zarr metadata — so this finds ANY
    committed leaf: root-MOC candidates first (window-qualified when
    given), the walk otherwise. A leaf from another window still serves —
    the schema is store-uniform (one manifest, one writer). Runs only on
    the already-exceptional empty path, so the extra stamp GETs (some
    repeating the caller's) stay off the hot path. ``None`` means the store
    has no stamped coverage at all — the caller's ``NoCoverageError`` case.
    """
    envelope = load_root_coverage(store_root, store=store)
    if envelope is not None:
        candidates = [
            leaf_path(int(w), window=window, path_grouping=path_grouping)
            for w in np.sort(ranges_words(envelope))
        ]
        stamps = read_commits(store_root, candidates, store=store, concurrency=concurrency)
        rel = next((r for r, s in zip(candidates, stamps) if s is not None), None)
        if rel is not None:
            return rel
    leaves = sorted(
        walk_leaves(store_root, store=store, concurrency=concurrency, path_grouping=path_grouping)
    )
    stamps = read_commits(store_root, leaves, store=store, concurrency=concurrency)
    return next((r for r, s in zip(leaves, stamps) if s is not None), None)


def open_hive(
    store_root: str,
    *,
    aoi=None,
    window: str | None = None,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    decode: bool = False,
    index_kind: str = "pandas",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    **store_kwargs: Any,
):
    """Open a morton-hive store as one xarray Dataset.

    Parameters
    ----------
    store_root : str
        Store root (local directory or ``s3://bucket/prefix``).
    aoi : array-like, optional
        Morton cover of the area of interest — packed ``uint64`` words or
        decimal strings, mixed orders allowed. Shards and rows outside the
        cover are excluded (rows exactly, via the ``morton`` coordinate).
    window : str, optional
        Window label for a time-windowed (``morton-hive/2``) store. Omitted
        on such a store, the error lists the labels that exist.
    anonymous : bool, optional
        Unsigned S3 requests (public buckets).
    fabricate_cell_ids : {"auto", True, False}, optional
        NESTED ``cell_ids`` posture (englacial/zagg#262: "NESTED is
        fabricated, never stored"). ``"auto"`` (default): a stored
        ``cell_ids`` coordinate is kept untouched; when absent (morton-only
        store) an exact NESTED view is fabricated from the ``morton``
        coordinate via :func:`moczarr.fabricate.fabricate_cell_ids`.
        ``True`` always fabricates (replacing any stored array — exact, so
        a no-op on dual-written stores); ``False`` never fabricates.
        Fabrication runs once post-concat on the final ``morton``
        coordinate — equivalent to per-leaf (the same words, and
        ``mort2healpix`` is elementwise) but a single vectorized call.
        The fabricated ``cell_ids`` is a Python-side convenience view: the
        dataset-level ``attrs["dggs"]`` block is left untouched, so on a
        morton-only store it still advertises the morton scheme while the
        added coordinate is NESTED. Re-serializing such a result is not
        internally consistent; the authoritative morton-only ``dggs``
        discriminator is owned by the zagg#262 convention work.
    decode : bool, optional
        Assign the xdggs ``MortonIndex`` to the ``morton`` coordinate before
        returning (``moczarr.dggs.decode``), enabling the ``ds.dggs``
        accessor. Requires the ``moczarr[xdggs]`` extra; the default leaves
        the result xdggs-free.
    index_kind : {"pandas", "moc"}, optional
        Index posture for the ``morton`` coordinate (the xdggs vocabulary).
        ``"pandas"`` (default) is the status quo: the stored coordinate is
        read and, with ``decode=True``, indexed through a ``PandasIndex``.
        ``"moc"`` is the lazy path: the on-disk ``morton``/``cell_ids``
        arrays are never read — the row domain comes from the same coverage
        arithmetic that selected the leaves (shard subtrees ∩ AOI), held as
        a :class:`moczarr.moc_index.MortonMocIndex` whose coordinate is
        fabricated on demand. The index attaches regardless of ``decode``
        (it is core, xarray-only); ``decode=True`` additionally wraps it for
        the ``ds.dggs`` accessor. Requiring ``decode`` here would chain the
        core lazy index to the xdggs extra, against the ratified placement.
    concurrency : int or None, optional
        Maximum in-flight metadata requests (the candidate leaves' stamp
        GETs, and the discovery walk's per-level LISTs), default 32 — the
        zarr-python knob vocabulary. ``None`` or ``1`` runs the serial path
        (debugging). Leaf DATA opens stay serial either way (issue #5,
        measured in the phase-3 bench; revisit with the lazy-index work).
    xr_kwargs : dict, optional
        Extra keyword arguments for each leaf's ``xarray.open_zarr`` (e.g.
        ``chunks={}`` for dask-backed laziness).
    **store_kwargs
        Extra keyword arguments for the object store (``region=...`` etc.).

    Returns
    -------
    xarray.Dataset
        Leaves concatenated along the cell dimension in ascending packed
        morton order, with ``morton``/``cell_ids`` as coordinates and the
        manifest summary under ``attrs["morton_hive"]``.

        An ``aoi`` (or ``window``) that intersects no coverage returns a
        schema-correct EMPTY dataset — every data variable and coordinate
        present with its stored name/dtype/attrs (schema read from one
        covered leaf's metadata) and zero rows along the cell dimension —
        and emits a ``UserWarning`` naming the store (issue #4). The empty
        result composes with ``decode``, ``index_kind="moc"`` (an empty
        interval domain), and ``fabricate_cell_ids``; ``xr.concat`` of the
        empty result with a non-empty one preserves dtypes — both sides
        carry every variable, so xarray fills nothing and no int→float NaN
        promotion occurs (pinned in ``tests/test_open.py``).

        Raises ``ValueError`` when the root is not a hive store (no
        manifest), and :class:`moczarr.NoCoverageError` — a ``ValueError``
        subclass — when the store has no stamped coverage anywhere: with
        zero committed leaves there is no schema source at all, whatever
        the query.
    """
    import xarray as xr
    from zarr.storage import ObjectStore

    if fabricate_cell_ids not in ("auto", True, False):
        raise ValueError(
            f"fabricate_cell_ids={fabricate_cell_ids!r}: expected 'auto', True, or False"
        )
    if fabricate_cell_ids != "auto":
        fabricate_cell_ids = bool(fabricate_cell_ids)
    if index_kind not in ("pandas", "moc"):
        raise ValueError(f"index_kind={index_kind!r}: expected 'pandas' or 'moc'")
    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    # ONE store construction pair for the whole open (issue #5): the obstore
    # handle serves every JSON/sidecar read; the zarr wrapper serves every
    # leaf open via deep paths through the parentless digit tree.
    obstore_store = open_object_store(store_root, **store_kwargs)
    zarr_store = ObjectStore(obstore_store, read_only=True)
    manifest = read_manifest(store_root, store=obstore_store)
    if manifest is None:
        raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    aoi_words = _aoi_words(aoi) if aoi is not None else None
    group = str(manifest["cell_order"])
    opened = []
    candidates = _candidate_leaves(
        store_root, manifest, aoi_words, window, store=obstore_store, concurrency=concurrency
    )
    stamps = read_commits(store_root, candidates, store=obstore_store, concurrency=concurrency)
    domain = None  # index_kind="moc": the accumulated interval-set row domain
    for rel, stamp in zip(candidates, stamps):
        if stamp is None:
            continue  # debris or a MOC-listed shard whose leaf is gone (D4)
        if aoi_words is not None:
            coverage = parse_leaf_coverage(stamp)
            if coverage is not None and coverage.get("box"):
                if box_and(coverage, aoi_words).size == 0:
                    continue  # conservative reject: false positives only
        if index_kind == "moc":
            # The leaf's row domain is arithmetic: zagg leaves are dense
            # within a shard, so rows = subtree ∩ AOI — the exact set the
            # pandas path's aoi_mask keeps, computed without reading the
            # stored coordinate (interval space, moczarr.ranges).
            from moczarr.ranges import MortonRanges

            shard, _label = split_leaf_name(rel.rsplit("/", 1)[-1])
            leaf_domain = MortonRanges.from_shards([morton_word(shard)], int(group))
            if aoi_words is not None:
                leaf_domain = leaf_domain.intersect(aoi_words)
                if leaf_domain.size == 0:
                    continue  # same skip the pandas path's empty aoi_mask takes
        ds = xr.open_zarr(
            zarr_store,
            group=f"{rel}/{group}",
            consolidated=False,
            zarr_format=3,
            **(xr_kwargs or {}),
        )
        coords = [name for name in ("morton", "cell_ids") if name in ds]
        ds = ds.set_coords(coords)
        if index_kind == "moc":
            # Drop the on-disk cell arrays before concat — lazily built, so
            # no chunk was read; the coordinate is the index's to fabricate.
            moc_dim = ds["morton"].dims[0] if "morton" in ds.coords else "cells"
            ds = ds.drop_vars(coords)
            if aoi_words is not None:
                full = MortonRanges.from_shards([morton_word(shard)], int(group))
                ds = ds.isel({moc_dim: full.rank(leaf_domain.fabricate())})
            domain = leaf_domain if domain is None else domain.union(leaf_domain)
        elif aoi_words is not None and "morton" in ds.coords:
            keep = aoi_mask(np.asarray(ds["morton"].values, dtype=np.uint64), aoi_words)
            if not keep.any():
                continue
            ds = ds.isel({ds["morton"].dims[0]: keep})
        opened.append(ds)
    if not opened:
        # Issue #4 contract: emptiness against a covered store is a data
        # answer — a schema-correct 0-row dataset plus a UserWarning. The
        # schema comes from ONE stamped leaf's metadata (an AOI-rejected
        # candidate when one exists, else _schema_leaf's store-wide search);
        # only a store with no stamped leaf anywhere has no schema to serve
        # and raises NoCoverageError.
        schema_rel = next((r for r, s in zip(candidates, stamps) if s is not None), None)
        schema_from_walk = False
        if schema_rel is None:
            schema_rel = _schema_leaf(
                store_root,
                window,
                store=obstore_store,
                concurrency=concurrency,
                path_grouping=manifest_path_grouping(manifest),
            )
            schema_from_walk = True
        if schema_rel is None:
            raise NoCoverageError(
                f"nothing to open at {store_root}: the store has no stamped coverage "
                f"anywhere (no committed leaf exists to define a schema)"
            )
        scope = [
            part
            for part, active in (
                ("the given AOI", aoi_words is not None),
                (f"window {window!r}", window is not None),
            )
            if active
        ]
        if not scope and schema_from_walk:
            # Unscoped whole-store open (aoi=None, window=None) where the root
            # MOC lists no openable leaf, yet _schema_leaf's walk found a
            # committed leaf on disk. That is a STALE root MOC, not an empty
            # store: silently returning 0 cells here (issue #4's empty
            # contract is scoped to an AOI/window over no coverage) would
            # hide committed data from a whole-store open. Raise instead —
            # never auto-walk the read path; regenerate the coverage
            # explicitly. Opting this case into the empty return is a
            # one-line change if that lean is preferred later.
            raise ValueError(
                f"stale root MOC at {store_root}: the root coverage lists no "
                f"openable leaf, but a committed leaf exists on disk at "
                f"{schema_rel!r}. Regenerate the root coverage before opening "
                f"(the store's writer / zagg's refresh_root_coverage / the "
                f"coverage sweep)."
            )
        # scope is non-empty here: the only unscoped way into this path is a
        # stale root MOC, handled by the raise above.
        warnings.warn(
            f"{' in '.join(scope)} intersects no coverage at {store_root}"
            "; returning a schema-correct empty dataset (0 cells)",
            UserWarning,
            stacklevel=2,
        )
        ds = xr.open_zarr(
            zarr_store,
            group=f"{schema_rel}/{group}",
            consolidated=False,
            zarr_format=3,
            **(xr_kwargs or {}),
        )
        coords = [name for name in ("morton", "cell_ids") if name in ds]
        ds = ds.set_coords(coords)
        empty_dim = ds["morton"].dims[0] if "morton" in ds.coords else "cells"
        if index_kind == "moc":
            from moczarr.ranges import MortonRanges

            moc_dim = empty_dim
            ds = ds.drop_vars(coords)
            domain = MortonRanges(np.empty((0, 2), dtype=np.uint64), int(group))
        opened.append(ds.isel({empty_dim: slice(0)}))
    dim = opened[0]["morton"].dims[0] if "morton" in opened[0].coords else "cells"
    if index_kind == "moc":
        dim = moc_dim
    result = xr.concat(opened, dim=dim) if len(opened) > 1 else opened[0]
    if index_kind == "moc":
        from moczarr.moc_index import MortonMocIndex

        assert domain is not None  # a leaf set it, or the empty path did
        index = MortonMocIndex(domain, dim=dim, name="morton")
        result = result.assign_coords(xr.Coordinates.from_xindex(index))
    if "morton" in result.coords and (
        fabricate_cell_ids is True
        or (fabricate_cell_ids == "auto" and "cell_ids" not in result.coords)
    ):
        ids = _fabricate_cell_ids(
            np.asarray(result["morton"].values, dtype=np.uint64),
            level=int(manifest["cell_order"]),
            # +1 frame vs a direct call so the >24 warning lands on the
            # user's open_hive(...) line, not this internal call site.
            _stacklevel=4,
        )
        result = result.assign_coords(cell_ids=(result["morton"].dims, ids))
    result.attrs["morton_hive"] = {
        k: manifest[k] for k in ("spec", "cell_order", "shard_order", "dataset")
    }
    if decode:
        from moczarr import dggs  # lazy: raises the pointed extra hint when absent

        result = dggs.decode(result, index_kind=index_kind)
    return result

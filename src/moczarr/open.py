"""``open_hive``: a morton-hive store as one lazy xarray Dataset.

The §5 reader flow, arithmetic-first (D10): manifest GET → root coverage
MOC ∩ AOI → hive paths by string arithmetic → stamped-leaf opens → concat
along the cell dimension. The discovery walk runs only when the root MOC is
absent/unusable (D9: caches degrade to the walk, never to wrong answers).
Debris (unstamped leaves) is skipped silently — absence of a stamp IS the
answer (D4).

AOI semantics: ``aoi`` is a morton cover — packed ``uint64`` words, decimal
strings, or an object exposing ``__morton_moc__()`` (mortie's ``Moc``),
mixed orders allowed. Shards are rejected arithmetically
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
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from moczarr.convention import (
    HIVE_SPEC_V2,
    decimal_order,
    leaf_path,
    manifest_path_grouping,
    morton_decimal,
    morton_word,
    parse_manifest,
    split_leaf_name,
    validate_window,
)
from moczarr.coverage import (
    aoi_mask,
    as_moc_words,
    as_toc_words,
    box_and,
    parse_leaf_coverage,
    ranges_words,
    root_coverage_and,
    temporal_keep,
)
from moczarr.exceptions import NoCoverageError
from moczarr.fabricate import fabricate_cell_ids as _fabricate_cell_ids
from moczarr.products import is_product_name, list_products, validate_product_name
from moczarr.store import (
    _stamp_from_meta,
    load_root_coverage,
    open_object_store,
    read_commits,
    read_leaf_metas,
    read_manifest,
    walk_leaves,
)


def _shard_leaf_name(rel: str) -> tuple[str, str | None] | None:
    """``(shard, window)`` for a walked ``*.zarr`` path, or ``None``.

    ``None`` means the object is not a shard leaf: overview zarrs at ancestor
    nodes reuse the window-naming dialect (``all.zarr`` / ``{window}.zarr``,
    zagg spec §4.2), so their stems are not morton decimals. The source walk
    skips them — the MOC-arithmetic path never names them at all (the root
    MOC is the *source* domain), and the walk must match it.
    """
    try:
        shard, label = split_leaf_name(rel.rsplit("/", 1)[-1])
        morton_word(shard)
    except ValueError:
        return None
    return shard, label


def candidate_leaves(
    store_root: str,
    manifest: dict,
    aoi=None,
    window: str | None = None,
    *,
    when=None,
    store: Any = None,
    concurrency: int | None = None,
) -> list[str]:
    """Store-relative leaf paths to try, ascending in packed-word order.

    The store's leaf-discovery seam, shared by :func:`open_hive` and
    :func:`moczarr.iter_occupancy_and` so the two agree on what a store
    contains by construction rather than by duplicated arithmetic. Public
    (issue #39) for readers that need the leaf roster without opening
    anything — the paths are what :func:`moczarr.open_leaf`,
    ``moczarr.store.read_commits`` and the ragged/HHDC readers take.

    Contract, in three parts:

    - **Discovery.** The root ``coverage.moc`` names the committed shards,
      and leaf paths come from it by string arithmetic (D10) — no LIST. The
      discovery walk (``moczarr.store.walk_leaves``) runs only when that
      envelope is absent or unusable, and is SEMANTICALLY EQUIVALENT: it is
      a fallback, never a wrong answer, so a caller never keys on which
      route ran (D9 — caches degrade to the walk, never to wrong answers).
      Both return CANDIDATES in ascending packed-word order, and the two
      candidate sets differ only where a stamp settles it anyway (D4:
      absence of a stamp IS the answer): the walk additionally names debris
      (unstamped leaves the root MOC does not list), and the arithmetic
      path additionally names MOC-listed shards whose object is gone. After
      the caller's commit-stamp GET — which every consumer here does — the
      two agree leaf-for-leaf with ``when=None``; under ``when=`` the walk
      returns a SUPERSET, because the temporal map it would prune against
      rides the same absent carrier (see the ``when`` bullet — degradation
      in the safe direction, D9, and no commit stamp settles it: a stamp
      says the leaf exists, not when its rows are). Objects that reuse the
      window-naming dialect
      at ancestor nodes (overview zarrs, spec §4.2) are not leaves and
      neither route returns them.
    - **Window selection.** ``window`` picks the time window of a
      ``morton-hive/2`` store, validated through the one
      ``convention.validate_window`` seam (which also refuses the reserved
      all-time token). A windowed store with ``window=None`` cannot take
      the arithmetic path — the walk enumerates what exists and the
      ``ValueError`` lists the labels present. ``window`` on an unwindowed
      store is a ``ValueError``.
    - **``aoi`` restriction (SHARD cover only).** A morton cover — packed
      words, decimal strings, or an object exposing ``__morton_moc__()``
      (mortie's ``Moc``), mixed orders allowed — restricts the
      returned leaves to those whose subtree meets it. It cuts SHARDS and
      nothing finer: a returned leaf is kept whole, and the cells inside it
      are NOT clipped to the cover, so any cell-level use of this roster is
      a superset with respect to the AOI (false positives possible, false
      negatives impossible — zagg's AOI-overhang convention). The exact
      cell-level cut is :func:`moczarr.coverage.aoi_mask`, which
      :func:`open_hive` applies to the ``morton`` coordinate itself (tier 2
      — the coordinate is the truth; the MOC tiers are only indexes).
    - **``when`` restriction (temporal, spec §10; issue #45).** A temporal
      query — ``(start, end)`` as ISO strings / ``datetime64`` / internal-ns
      ints (a CLOSED real interval — unlike every other window in this
      docstring, ``end`` itself is included), raw ``uint64`` toc words, or
      an object exposing ``__toc_words__()`` (mortie's ``Toc``) — prunes
      candidates against the
      root envelope's tier-1 per-shard toc word map, before anything is
      opened. The §10 asymmetry is the contract: only a shard the map LISTS
      whose word fails ``mortie.toc_overlaps`` for the window is dropped; a
      shard ABSENT from the map is always KEPT (unlisted is *unknown*, never
      *empty* — the section is a regenerable accelerator, not truth). A
      store publishing no ``temporal`` section prunes nothing, silently
      (§10's absence rule), and ``when=None`` is byte-identical to today.
      Pruning rides the root envelope's tier-1 map ALONE, so the discovery
      walk — which runs exactly when that envelope is absent or unusable —
      has nothing to prune against and returns the unpruned candidate set,
      a conservative superset (D9: degrade, never wrong).
      Like ``aoi``, the cut is conservative and SHARD-level: kept leaves may
      hold rows outside the window. The over-report near window edges runs
      to a few seconds — up to TWO encoding quanta, one from round-tripping
      the query window itself through ``span2toc``/``toc2time`` and one from
      the shard's own outward-rounded word under ``mortie.toc_overlaps`` —
      and every widening is outward, so it never under-reports.
    """
    if aoi is not None:
        aoi = as_moc_words(aoi)
    when_words = as_toc_words(when) if when is not None else None
    windowed = manifest["spec"] == HIVE_SPEC_V2
    grouping = manifest_path_grouping(manifest)
    if window is not None:
        # The one window-validation seam every non-overview entry point runs
        # through: open_hive directly, open_store via its per-product call.
        # It is also what refuses the reserved all-time token (#30).
        validate_window(window, where=store_root)
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
        if when_words is not None and words.size:
            words = words[temporal_keep(words, envelope, when_words)]
        return [leaf_path(int(w), window=window, path_grouping=grouping) for w in np.sort(words)]
    # Walk fallback (no usable root MOC), and the windowed-discovery case.
    found: dict[str, int] = {}
    labels: set[str] = set()
    for rel in walk_leaves(
        store_root, store=store, concurrency=concurrency, path_grouping=grouping
    ):
        named = _shard_leaf_name(rel)
        if named is None:
            continue  # an overview object at an ancestor node, not a leaf
        shard, label = named
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
    # No temporal pruning here by construction: reaching the walk means the
    # root envelope is absent or unusable, so there is no tier-1 map to
    # prune against and the candidate set stays an unpruned superset.
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
        rel
        for rel in walk_leaves(
            store_root, store=store, concurrency=concurrency, path_grouping=path_grouping
        )
        if _shard_leaf_name(rel) is not None  # overview objects serve no leaf schema
    )
    stamps = read_commits(store_root, leaves, store=store, concurrency=concurrency)
    return next((r for r, s in zip(leaves, stamps) if s is not None), None)


def _check_composition_fill(ds, rel: str) -> None:
    """Refuse a composition array whose declared ``fill_value`` is not 0 (§3).

    ``zagg-composition/1`` packs an empty signal stratum to ``0``, and spec §3
    **MUST**s the array's fill value to be the same ``0`` so an *unwritten*
    cell decodes to the same "no flags occurred" word. A nonzero fill makes
    every unwritten cell report spurious lane presence — a fill of 1 reads as
    "``land`` occurred exactly" — which is a wrong answer, not a degraded one,
    so it raises rather than warning (the D9 line; contrast the tolerant
    coverage caches). zagg enforces the same rule writer-side at config
    validation, so a store tripping this is malformed, not merely old.

    Costs nothing: both the attrs and the declared fill come from the leaf
    metadata ``xarray`` has just read, so there is no extra object GET. The
    lane algebra stays pure — :mod:`moczarr.composition` takes words and
    attrs and never sees a ``zarr.Array``, which is why this check lives on
    the open path instead.
    """
    for name, var in ds.variables.items():
        block = var.attrs.get("composition")
        if not isinstance(block, Mapping) or "spec" not in block:
            continue
        fill = var.encoding.get("fill_value")
        if fill is None or fill != 0:
            raise ValueError(
                f"non-conforming composition array {name!r} at {rel}: spec §3 requires "
                f"fill_value 0, so that an unwritten cell decodes to the same word an "
                f"empty signal stratum packs to, but this array declares {fill!r} — "
                f"every unwritten cell would report spurious lane presence"
            )


def open_hive(
    store_root: str,
    *,
    product: str | None = None,
    aoi=None,
    window: str | None = None,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    decode: bool = False,
    index_kind: str = "moc",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    _objects_out: list | None = None,
    **store_kwargs: Any,
):
    """Open a morton-hive store as one xarray Dataset.

    Parameters
    ----------
    store_root : str
        Store root (local directory or ``s3://bucket/prefix``) — a bare
        single-product store, or a multi-product store root when ``product``
        is given.
    product : str, optional
        Named product of a multi-product store root (D19, spec §6.5): sugar
        for opening the ``{store_root}/{product}`` subtree, which is a
        complete morton-hive store in its own right. Omitted on a
        multi-product root, the error lists the product names that exist
        (see :func:`moczarr.products.list_products`). Single-product stores
        are unaffected — the bare form is the valid degenerate case.
    aoi : array-like, optional
        Morton cover of the area of interest — packed ``uint64`` words,
        decimal strings, or an object exposing ``__morton_moc__()``
        (mortie's ``Moc``), mixed orders allowed. Shards and rows outside
        the cover are excluded (rows exactly, via the ``morton`` coordinate).
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
    index_kind : {"moc", "pandas"}, optional
        Index posture for the ``morton`` coordinate (the xdggs vocabulary).
        ``"moc"`` (default) is the lazy path: the on-disk
        ``morton``/``cell_ids`` arrays are never read — the row domain comes
        from the same coverage arithmetic that selected the leaves (shard
        subtrees ∩ AOI), held as a
        :class:`moczarr.moc_index.MortonMocIndex` whose coordinate is
        fabricated on demand. The index attaches regardless of ``decode``
        (it is core, xarray-only); ``decode=True`` additionally wraps it for
        the ``ds.dggs`` accessor. Requiring ``decode`` here would chain the
        core lazy index to the xdggs extra, against the ratified placement.
        ``"pandas"`` materializes instead: the stored coordinate is read
        and, with ``decode=True``, indexed through a ``PandasIndex`` — use
        it when a workflow needs what the interval index cannot represent
        (notably ``xr.concat`` of overlapping or out-of-order domains; the
        disjoint, ascending batch-sweep concat works on the moc default).
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
        interval domain), and ``fabricate_cell_ids``. ``xr.concat`` of the
        empty result with a non-empty one preserves dtypes on either
        ``index_kind`` — both sides carry every variable, so xarray fills
        nothing and no int→float NaN promotion occurs; on the moc default
        the empty side contributes no intervals, so the concat returns the
        non-empty domain unchanged (issue #4's empty-composes-through-concat
        contract, pinned in ``tests/test_open.py``).

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
    if product is not None:
        # D19 sugar: a product subtree IS a complete morton-hive store, so
        # the whole open simply re-roots on it (equivalent to opening the
        # subtree path directly, pinned by a test).
        from moczarr.products import validate_product_name

        validate_product_name(product)
        store_root = f"{store_root.rstrip('/')}/{product}"
    # ONE store construction pair for the whole open (issue #5): the obstore
    # handle serves every JSON/sidecar read; the zarr wrapper serves every
    # leaf open via deep paths through the parentless digit tree.
    obstore_store = open_object_store(store_root, **store_kwargs)
    zarr_store = ObjectStore(obstore_store, read_only=True)
    manifest = read_manifest(store_root, store=obstore_store)
    if manifest is None:
        if product is None:
            # A multi-product root has no root manifest by design (§6.5
            # content discrimination) — probe for named products so the
            # error is pointed, not "not a hive store". Error path only.
            from moczarr.products import list_products

            names = [p["name"] for p in list_products(store_root, store=obstore_store)]
            if names:
                raise ValueError(
                    f"{store_root} is a multi-product store root (products: {names}); "
                    f"pass product=... to open one (D19, mortie spec §6.5)"
                )
        raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    aoi_words = as_moc_words(aoi) if aoi is not None else None
    group = str(manifest["cell_order"])
    opened = []
    candidates = candidate_leaves(
        store_root, manifest, aoi_words, window, store=obstore_store, concurrency=concurrency
    )
    # One zarr.json GET per candidate serves BOTH the commit stamp and — for
    # the tree layer's per-object role surfacing (_objects_out, a private
    # collection hook in the zagg occupied_out style) — the D11 role attrs.
    metas = read_leaf_metas(store_root, candidates, store=obstore_store, concurrency=concurrency)
    stamps = [_stamp_from_meta(meta) for meta in metas]
    domain = None  # index_kind="moc": the accumulated interval-set row domain
    for rel, meta, stamp in zip(candidates, metas, stamps):
        if stamp is None:
            continue  # debris or a MOC-listed shard whose leaf is gone (D4)
        if _objects_out is not None:
            # Per-object role entry for every leaf this open ADMITS, recorded
            # before the AOI rejects below: the roster answers structural
            # questions (moczarr.pyramid.source_orders), so it must not shrink
            # with the query. Role absence means source (zagg spec §4.3); a
            # role-carrying object inside the source domain is off-convention
            # but surfaced verbatim rather than hidden (mixture-tolerant).
            attrs = meta.get("attributes") if isinstance(meta, dict) else None
            node_dec, node_label = split_leaf_name(rel.rsplit("/", 1)[-1])
            _objects_out.append(
                {
                    "node": node_dec,
                    "window": node_label,
                    "role": (attrs or {}).get("role") or "source",
                }
            )
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
        _check_composition_fill(ds, rel)
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
        _check_composition_fill(ds, schema_rel)
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
    # Data-variable order is otherwise the completion order of zarr-python's
    # async member listing — non-deterministic run to run (a fresh open
    # reshuffles ds.data_vars). Sort lexically so a given store always opens
    # with the same variable order: reproducible reprs and stable notebook
    # outputs. Coordinates and the morton index ride along unchanged.
    result = result[sorted(result.data_vars)]
    result.attrs["morton_hive"] = {
        k: manifest[k] for k in ("spec", "cell_order", "shard_order", "dataset")
    }
    if decode:
        from moczarr import dggs  # lazy: raises the pointed extra hint when absent

        result = dggs.decode(result, index_kind=index_kind)
    return result


def open_leaf(
    store_root: str,
    shard: str | int,
    *,
    window: str | None = None,
    product: str | None = None,
    manifest: dict | None = None,
    anonymous: bool = False,
    store: Any = None,
    **store_kwargs: Any,
):
    """Open ONE shard's leaf zarr as a read-only zarr store.

    The leaf-direct twin of :func:`open_hive`, for the per-leaf readers
    (:func:`moczarr.hhdc.read_tensors`, :func:`moczarr.ragged.open_ragged`,
    ...): it owns the three layers a caller otherwise hand-assembles — the
    leaf path from the manifest's grammar (``convention.leaf_path`` under the
    manifest's ``path_grouping``), the transport with the SAME credential /
    ``anonymous`` handling as :func:`open_hive`, and the
    ``zarr.storage.ObjectStore`` read-only wrapper the readers take. No
    external path arithmetic, no bare-obstore incantation.

    Parameters
    ----------
    store_root : str
        The hive store root (``s3://...`` or a local directory).
    shard : str or int
        The leaf's shard — packed morton word or decimal string.
    window : str, optional
        Time-window label for a windowed leaf (``{id}_{window}.zarr``).
        Required on a windowed (``morton-hive/2``) store and refused on an
        unwindowed one, matching :func:`open_hive`; the reserved all-time
        token ``all`` is refused at the same seam (espg/moczarr#30).
    product : str, optional
        Product name under a multi-product root (D19) — the open re-roots on
        the product subtree, exactly as :func:`open_hive` does.
    manifest : dict, optional
        An already-read manifest **of the subtree actually opened** — the
        PRODUCT's manifest when ``product`` is given, not the root's.
        Passing it skips this call's manifest GET (the iterate-many-leaves
        case reads it once and threads it here) and is validated through
        :func:`moczarr.convention.parse_manifest` like any read one.
    anonymous : bool
        Skip request signing (public buckets).
    store : obstore store, optional
        A shared handle for the manifest GET (issue #5), rooted at the
        subtree this call opens — the PRODUCT subtree when ``product`` is
        given. The returned leaf store is always a fresh, leaf-rooted open;
        only the manifest read can share a handle.
    **store_kwargs
        Forwarded to :func:`moczarr.store.open_object_store`
        (``region=...``, explicit keys, ...).

    Returns
    -------
    zarr.storage.ObjectStore
        Read-only store rooted at the leaf; pass it with a field path like
        ``"{cell_order}/{name}"`` to the per-leaf readers. Deliberately bare
        — it carries no manifest — so a caller that needs ``cell_order`` for
        those paths takes it from ``manifest=`` (the one it threaded) or
        from :func:`moczarr.store.read_manifest`.

    Raises
    ------
    ValueError
        When ``store_root`` (or the product subtree) has no manifest — the
        path grammar (``path_grouping``) is manifest-declared, so a leaf
        path cannot be derived without one — when ``window`` disagrees with
        the manifest's spec, and when ``shard`` is not at the manifest's
        ``shard_order`` (a cell id where a shard id belongs).
    """
    from zarr.storage import ObjectStore

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    if product is not None:
        from moczarr.products import validate_product_name

        validate_product_name(product)
        store_root = f"{store_root.rstrip('/')}/{product}"
    if window is not None:
        # The same window seam every non-overview entry point runs through;
        # it is also what refuses the reserved all-time token (#30).
        validate_window(window, where=store_root)
    if manifest is None:
        manifest = read_manifest(store_root, store=store, **store_kwargs)
        if manifest is None:
            if product is None:
                # A multi-product root has no root manifest by design (§6.5);
                # probe so the error is pointed, exactly as open_hive's is.
                names = [p["name"] for p in list_products(store_root, store=store, **store_kwargs)]
                if names:
                    raise ValueError(
                        f"{store_root} is a multi-product store root (products: {names}); "
                        f"pass product=... to open one (D19, mortie spec §6.5)"
                    )
            raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    else:
        manifest = parse_manifest(manifest)
    windowed = manifest["spec"] == HIVE_SPEC_V2
    if window is not None and not windowed:
        raise ValueError(
            f"window={window!r} on a {manifest['spec']} store: unwindowed stores "
            f"have no window leaves (schedule: none)"
        )
    if window is None and windowed:
        # open_hive lists the labels it walked; a leaf-direct open names ONE
        # object and never enumerates, so it says what it needs instead.
        raise ValueError(
            f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; pass window=... "
            f"(a leaf-direct open names one object, so there is nothing to enumerate)"
        )
    rel = leaf_path(shard, window, path_grouping=manifest_path_grouping(manifest))
    order = decimal_order(morton_decimal(shard))
    if order != int(manifest["shard_order"]):
        raise ValueError(
            f"shard {shard!r} is an order-{order} id, but {store_root} shards at order "
            f"{manifest['shard_order']} — leaves are named by SHARD id (a cell id "
            f"names no leaf)"
        )
    leaf_root = f"{store_root.rstrip('/')}/{rel}"
    return ObjectStore(open_object_store(leaf_root, **store_kwargs), read_only=True)


def _degenerate_node_name(manifest: dict) -> str:
    """The one child's node name for a bare single-product store.

    A bare store has no ``{name}/`` prefix to name its node, so the name
    derives from the manifest's dataset ``short_name`` lowercased when that
    yields a legal §6.5 product name (``"ATL06"`` → ``"atl06"``), else the
    fixed fallback ``"product"``. Deterministic and content-derived — never
    the store-root basename, which changes under copies and mounts.
    """
    dataset = manifest.get("dataset") or {}
    name = str(dataset.get("short_name") or "").lower()
    return name if is_product_name(name) else "product"


def open_store(
    store_root: str,
    *,
    products: Sequence[str] | None = None,
    aoi=None,
    window: str | None = None,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    decode: bool = False,
    index_kind: str = "moc",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    **store_kwargs: Any,
):
    """Open a store root — multi-product or bare — as one ``xarray.DataTree``.

    The tree shape follows the zagg O14 rulings (issue #15): the **product
    axis becomes child nodes** — one node per named product (D19, mortie
    spec §6.5), each holding exactly the lazy Dataset :func:`open_hive`
    returns for that product — and the root node is empty (store-level
    attrs only; a multi-product root is manifest-less by design). The
    window axis is **never** nodes (time stays a per-node dimension,
    scoped by ``window=``), and the spatial digit axis is never nodes.

    **Resolution (pyramid-order) nodes** (issue #15 phase 8b): when a
    product's manifest declares sweep-generated overviews (the zagg spec
    §4.5 ``pyramid.overview.orders`` binding — manifest declaration is the
    only discovery path), the product node becomes an empty intermediate
    and the stored orders become its children, named by the integer cell
    order they store: the source data is the source-order child (the same
    ``open_hive`` Dataset, plus per-object role entries under
    ``attrs["zagg_objects"]``) and each declared overview order with at
    least one stamped object is a sibling node opened by
    :func:`moczarr.pyramid.open_overview_order`. ``role`` is per object,
    never per node; sibling variable sets differ in both directions; the
    seamless mixed-order composite is a *computed* view, never a node.
    Selection helpers: :func:`moczarr.pyramid.source_orders`,
    :func:`moczarr.pyramid.overview_orders`,
    :func:`moczarr.pyramid.finest_source_at`.

    Named after its scope (the store), not its return type: the xarray-
    native name ``open_datatree`` is reserved for zarr-native hierarchies,
    which the hive tree deliberately is not (the D12 interop hierarchy is
    ``xr.open_datatree``'s to open someday).

    Parameters
    ----------
    store_root : str
        Store root (local directory or ``s3://bucket/prefix``): a
        multi-product root (named product children, no root manifest) or a
        bare single-product store (manifest at the root), discriminated by
        content per spec §6.5.
    products : sequence of str, optional
        Open only these products (a filter for stores with many). An empty
        sequence raises — a childless tree is never the intended answer;
        pass ``None`` (the default) for "all". Names are
        checked against the spec §6.5 grammar **before any I/O**; unknown
        (but well-formed) names raise once the roster is known, listing what
        exists. Children stay in name-sorted order regardless of the
        filter's order.
    aoi, window, anonymous, fabricate_cell_ids, decode, index_kind, \
concurrency, xr_kwargs, **store_kwargs
        Forwarded to each node's :func:`open_hive` (see there), with one
        tree-level refinement: ``window`` is forwarded only to the
        time-windowed (``morton-hive/2``) products — an unwindowed
        product's node holds the whole product (it has no window leaves to
        select), so one call opens a store that mixes windowed and
        unwindowed products. A ``window`` against a tree with *no*
        windowed product raises, preserving :func:`open_hive`'s
        typo-catching posture. Omitting ``window`` while a windowed
        product is selected raises per-product with the labels that exist
        (scope with ``products=`` to skip windowed products instead).

    Returns
    -------
    xarray.DataTree
        Root node: no data variables, no coordinates — only
        ``attrs["morton_hive_store"]`` (``store_root`` plus the name-sorted
        ``products`` roster the store *publishes*, unfiltered — exactly
        :func:`list_products`, so it is ``[]`` for a bare store). One child node
        per opened product, named by its product name; each child is the
        per-node ``open_hive`` Dataset unchanged (laziness, index posture,
        and the issue-#4 empty-AOI contract all apply per node), plus
        ``attrs["semantic_hash"]`` when the product's manifest carries the
        D19 identity hash. Sibling schemas may differ freely — the
        non-alignable-groups case is exactly what the tree is for.

        A **bare single-product store** returns the valid degenerate form:
        a one-child tree, the child named from the manifest's dataset
        ``short_name`` lowercased (fallback ``"product"``). That name is
        *derived*, not a published product, so the roster stays ``[]`` and
        the store attrs carry ``bare: True`` plus ``node: <derived name>``
        instead of fabricating a roster ``list_products`` would deny —
        and ``products=`` against a bare store raises rather than filtering
        on a name the store never published.

        Raises ``ValueError`` when the root has neither products nor a
        manifest, and per-node errors (``NoCoverageError``, the windowed
        ``window=`` requirement) propagate unchanged.
    """
    import xarray as xr

    # Grammar before I/O: an off-spec product name is a caller bug, so it
    # costs no LIST/GET to reject (spec §6.5 names are pure syntax). An
    # empty filter is loud too — a childless tree is the module's one silent
    # nonsense answer, and in practice it means the caller's filter list
    # came out empty.
    wanted = {validate_product_name(name) for name in products} if products is not None else None
    if wanted is not None and not wanted:
        raise ValueError(
            f"products=[] selects nothing at {store_root} — omit products= to open every product"
        )

    obstore_store = open_object_store(store_root, anonymous=anonymous, **store_kwargs)
    records = list_products(store_root, store=obstore_store)
    bare = not records
    if bare:
        manifest = read_manifest(store_root, store=obstore_store)
        if manifest is None:
            raise ValueError(
                f"no products and no morton_hive.json at {store_root} — not a hive store root"
            )
        records = [
            {
                "name": _degenerate_node_name(manifest),
                "spec": manifest["spec"],
                "semantic_hash": manifest.get("semantic_hash"),
                "manifest": manifest,
            }
        ]
    # The roster is what the store *publishes* — exactly ``list_products``,
    # empty for a bare store; the degenerate child's name is derived from
    # the manifest, so it is reported separately and never as a product.
    roster = [] if bare else [record["name"] for record in records]
    if wanted is not None:
        if bare:
            raise ValueError(
                f"products={sorted(wanted)} but {store_root} is a bare single-product store "
                "(no named products — spec §6.5 product prefixes); open it without products="
            )
        missing = sorted(wanted - set(roster))
        if missing:
            raise ValueError(
                f"products {missing} not found at {store_root} (products present: {roster})"
            )
        records = [record for record in records if record["name"] in wanted]
    if window is not None:
        # Ahead of the typo-catching check below on purpose: a user who passes
        # the reserved all-time token gets told that, not "no windowed product
        # is selected", whatever the roster happens to be (#30).
        validate_window(window, where=store_root)
    if window is not None and all(record["spec"] != HIVE_SPEC_V2 for record in records):
        raise ValueError(
            f"window={window!r} but no windowed ({HIVE_SPEC_V2}) product is selected at "
            f"{store_root} (specs: { {r['name']: r['spec'] for r in records} })"
        )
    from moczarr import pyramid

    nodes: dict[str, Any] = {}
    for record in records:
        name = record["name"]
        node_window = window if record["spec"] == HIVE_SPEC_V2 else None
        manifest_rec = record.get("manifest")
        # Node discovery is MANIFEST DECLARATION (zagg spec §4.5; issue #15
        # phase 8b): the reader binds pyramid.overview.orders and nothing
        # else. Empty/absent -> today's flat product node, unchanged.
        cell_orders = pyramid.overview_cell_orders(manifest_rec) if manifest_rec else {}
        objects: list[dict] = []
        ds = open_hive(
            store_root,
            product=None if bare else name,
            aoi=aoi,
            window=node_window,
            anonymous=anonymous,
            fabricate_cell_ids=fabricate_cell_ids,
            decode=decode,
            index_kind=index_kind,
            concurrency=concurrency,
            xr_kwargs=xr_kwargs,
            _objects_out=objects if cell_orders else None,
            **store_kwargs,
        )
        if not cell_orders or manifest_rec is None:  # None narrows for mypy
            if record.get("semantic_hash"):
                ds.attrs["semantic_hash"] = record["semantic_hash"]
            nodes[name] = ds
            continue
        # Pyramid form ({product}/{order}, zagg#201's ratified layout): the
        # product node becomes an empty intermediate carrying the product's
        # identity attrs; the source data becomes the source-order child
        # (named by the cell order it stores) with its per-object role
        # entries; each declared overview order that carries at least one
        # stamped object becomes a sibling order node. The seamless
        # mixed-order composite is a COMPUTED view, never a node.
        ds.attrs[pyramid.OBJECTS_ATTR] = objects
        product_attrs: dict[str, Any] = {
            "morton_hive": {
                key: manifest_rec[key] for key in ("spec", "cell_order", "shard_order", "dataset")
            }
        }
        if record.get("semantic_hash"):
            product_attrs["semantic_hash"] = record["semantic_hash"]
        nodes[name] = xr.Dataset(attrs=product_attrs)
        nodes[f"{name}/{int(manifest_rec['cell_order'])}"] = ds
        product_root = store_root if bare else f"{store_root.rstrip('/')}/{name}"
        # ONE store construction and ONE root-MOC read for this product's whole
        # order level (issue #5): the sidecar tier must not grow with the number
        # of declared orders — on a 5-order pyramid over S3 that would be 5
        # round-trips for one per-product envelope. A None envelope means "not
        # supplied", so the degraded no-MOC case re-reads and warns per order
        # exactly as before.
        product_store = open_object_store(product_root, anonymous=anonymous, **store_kwargs)
        product_envelope = load_root_coverage(product_root, store=product_store)
        for ancestor_order, target_order in cell_orders.items():
            overview_ds = pyramid.open_overview_order(
                product_root,
                manifest_rec,
                ancestor_order,
                aoi=aoi,
                window=node_window,
                fabricate_cell_ids=fabricate_cell_ids,
                decode=decode,
                index_kind=index_kind,
                concurrency=concurrency,
                xr_kwargs=xr_kwargs,
                store=product_store,
                _envelope=product_envelope,
            )
            if overview_ds is None:
                continue  # open_overview_order warned; the node is omitted
            nodes[f"{name}/{target_order}"] = overview_ds
    store_attrs: dict[str, Any] = {"store_root": store_root, "products": roster}
    if bare:
        store_attrs["bare"] = True
        store_attrs["node"] = records[0]["name"]
    root = xr.Dataset(attrs={"morton_hive_store": store_attrs})
    return xr.DataTree.from_dict({"/": root, **nodes})

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
from collections.abc import Mapping, Sequence
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
from moczarr.products import is_product_name, list_products, validate_product_name
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
    Resolution (pyramid-order) nodes are a designed, not-yet-implemented
    second level — gated on zagg#201's first overview fixture; see the
    concepts page's design section.

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
    if window is not None and all(record["spec"] != HIVE_SPEC_V2 for record in records):
        raise ValueError(
            f"window={window!r} but no windowed ({HIVE_SPEC_V2}) product is selected at "
            f"{store_root} (specs: { {r['name']: r['spec'] for r in records} })"
        )
    nodes: dict[str, Any] = {}
    for record in records:
        ds = open_hive(
            store_root,
            product=None if bare else record["name"],
            aoi=aoi,
            window=window if record["spec"] == HIVE_SPEC_V2 else None,
            anonymous=anonymous,
            fabricate_cell_ids=fabricate_cell_ids,
            decode=decode,
            index_kind=index_kind,
            concurrency=concurrency,
            xr_kwargs=xr_kwargs,
            **store_kwargs,
        )
        if record.get("semantic_hash"):
            ds.attrs["semantic_hash"] = record["semantic_hash"]
        nodes[record["name"]] = ds
    store_attrs: dict[str, Any] = {"store_root": store_root, "products": roster}
    if bare:
        store_attrs["bare"] = True
        store_attrs["node"] = records[0]["name"]
    root = xr.Dataset(attrs={"morton_hive_store": store_attrs})
    return xr.DataTree.from_dict({"/": root, **nodes})

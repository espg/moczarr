"""One resolution level of a pyramid store as ONE xarray Dataset (issue #37).

The per-order data model the #37 thread ruled ("one dataset per order" —
espg, 2026-08-05): whatever artifact kind materializes a resolution — the
source leaves at native order, a §4.6 leaf column's coarser groups, or a
§4.1 overview artifact at an ancestor node — a reader gets the SAME
in-memory shape, one :class:`xarray.Dataset` per cell order:

- **dims**: the 1-D ``cells`` axis, nested-ascending in packed-word order;
- **coords**: ``morton`` — the packed ``uint64`` cell words at this level's
  cell order, the native coordinate (identity is morton, never lat/lon;
  geometry stays derivable via mortie), lazy under the default
  ``index_kind="moc"`` (:class:`moczarr.moc_index.MortonMocIndex`) — plus
  the fabricated NESTED ``cell_ids`` view;
- **data variables**: dense fields as their stored dtypes; ragged digest
  fields ride as their ENCODED ``zagg-ragged/1`` vlen-bytes variables
  (dtype ``object``) on the same ``cells`` axis, the §1.2 ``ragged`` attrs
  block verbatim on the variable — decode is
  :func:`moczarr.ragged.parse_ragged_attrs` +
  :func:`moczarr.ragged.decode_cell` on pulled values, or the
  store-addressed :func:`moczarr.ragged.read_ragged`. Eager decode into
  padded dense tensors is rejected (it breaks laziness and invents a fill
  the spec does not have), and dropping ragged fields is rejected too (the
  level would stop being the store's level);
- **attrs**: ``morton_hive`` (the level summary — ``cell_order`` is THIS
  level's resolution, ``shard_order`` the artifact node order) and
  ``zagg_objects`` (the per-artifact roster: each entry carries the
  object's own provenance block verbatim — ``zagg_overview`` for overview
  artifacts, ``zagg_column`` for columns — so fold regime,
  ``merges_from_raw``, ``source_children``, and stamped ``demotions``
  records all ride along without re-keying). The roster is STRUCTURAL —
  recorded for every stamped object before any AOI filtering — on the
  source and overview arms; on the §4.6 column arm it is QUERY-SCOPED, a
  named exception :func:`open_column_order` documents.

The three artifact kinds keep their existing openers — ``open_hive`` for
the native order, :func:`moczarr.pyramid.open_overview_order` for ancestor
artifacts — and this module adds the missing third
(:func:`open_column_order`: one Dataset per column-carried resolution,
assembled across the covered leaves' §4.6 columns) plus the resolution
table (:func:`pyramid_levels`) that says which opener owns which cell
order. On top of the per-level block sits the espg/moczarr#36b multi-order
ASSEMBLY, :func:`open_pyramid`: one ``xarray.DataTree`` whose groups are
the store's resolution levels — each group the per-level Dataset — with
the normalized declaration record on the root.

Native surfaces only (the englacial/zagg#550 ruling): nothing in this
module touches ``moczarr.dggs`` or xdggs, and the new entry points expose
no ``decode=``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from moczarr.column import COLUMN_ATTR, COLUMN_SPEC
from moczarr.convention import (
    ALL_TOKEN,
    HIVE_SPEC_V2,
    column_path,
    manifest_path_grouping,
    morton_decimal,
    morton_word,
    validate_window,
)
from moczarr.coverage import ranges_words, root_coverage_and
from moczarr.pyramid import ROLE_ATTR, _skip, pyramid_declaration
from moczarr.ranges import MortonRanges
from moczarr.store import _stamp_from_meta, load_root_coverage, read_leaf_metas

#: Dataset-attrs key carrying one level's record (:func:`open_level`).
LEVEL_ATTR = "zagg_level"


def pyramid_levels(manifest: dict) -> dict[int, dict]:
    """``{cell order: level record}`` — every addressable resolution, finest first.

    The resolution table of the #37 model: one entry per level of the
    store, keyed by the CELL ORDER a reader picks (never by node order —
    resolution is the reader-facing axis, zagg spec §4's "a reader picks
    its resolution and reads it"). Each record is::

        {"cell_order": r, "order": k, "artifact": kind}

    where ``order`` is the hive-tree node order whose artifact carries the
    level and ``artifact`` names the kind — the vocabulary deliberately
    mirrors the ``zagg-multiscales/1`` dataset entries (zagg#392/#555:
    ``order``/``cells``/``artifact``), read here from the normative
    ``pyramid`` block via :func:`moczarr.pyramid.pyramid_declaration`
    rather than the derived mirror:

    - ``"source"`` — the native resolution (``manifest["cell_order"]`` at
      the shard nodes), present for EVERY store: a declared-off pyramid is
      the one-level table;
    - ``"overview"`` — a §4.1 ancestor-node artifact: the ``/1``
      constant-depth orders, or a ``/2`` fixed-ladder rung;
    - ``"column"`` — a ``/2`` declared leaf resolution, materialized by the
      §4.6 leaf columns (``leaf_cells``; ``/1`` declares none).

    Declared, not probed: what is on disk is :func:`moczarr.pyramid.
    read_pyramid`'s question (declaring is free, sweeping is the
    operational decision — zagg#381 point (11)), and the §4.6 node-order
    member (``r == shard_order`` — a recorded column group, never a
    manifest member) is deliberately absent here while
    :func:`open_column_order` still opens it. Raises when one cell order is
    declared at two nodes: "one resolution names one level" is this model's
    addressing law, and a duplicated resolution has no single answer.
    """
    cell_order = int(manifest["cell_order"])
    shard_order = int(manifest["shard_order"])
    levels: dict[int, dict] = {
        cell_order: {"cell_order": cell_order, "order": shard_order, "artifact": "source"}
    }
    decl = pyramid_declaration(manifest)
    if decl is None:
        return levels
    for r in decl["leaf_cells"] or []:
        levels[int(r)] = {"cell_order": int(r), "order": shard_order, "artifact": "column"}
    for k, cells in decl["cell_orders"].items():
        for r in cells:
            if int(r) in levels:
                raise ValueError(
                    f"cell order {r} is declared at two levels (node {k} and "
                    f"{levels[int(r)]['artifact']!r} at node {levels[int(r)]['order']}): "
                    f"one resolution names one level in this model, so a duplicated "
                    f"resolution has no single answer (issue #37)"
                )
            levels[int(r)] = {"cell_order": int(r), "order": int(k), "artifact": "overview"}
    return dict(sorted(levels.items(), reverse=True))


def _column_entry(attrs: dict, decimal: str, window: str | None, shard_order: int) -> dict | None:
    """One stamped column's per-object entry, or ``None`` to drop the object.

    The column twin of :func:`moczarr.pyramid._object_entry`, with the same
    two severities (§4.6 columns are derived artifacts a reader MUST NOT
    require, like §4.1 overviews):

    * **Uninterpretable** — a role other than ``"column"`` at the name
      seam, a missing :data:`moczarr.column.COLUMN_ATTR` block, an unknown
      revision. Warn and skip: one malformed artifact must not take the
      level's other columns down. (Contrast
      :func:`moczarr.column.read_column_record`, which names ONE object the
      caller asked about and raises — a point query cannot degrade by
      omission.)
    * **Interpretable but wrong** — a block positively declaring another
      leaf's ``node``/``order``, or a ``window`` the basename does not
      round-trip with. Raises: those groups would be read under the wrong
      node's identity — a wrong answer, not a missing one.
    """
    role = attrs.get(ROLE_ATTR)
    block = attrs.get(COLUMN_ATTR)
    if role != "column":
        _skip(
            decimal,
            f"is stamped at the §4.6 column basename but declares role {role!r}, "
            f"not 'column' (zagg spec §4.6)",
        )
        return None
    if not isinstance(block, dict):
        _skip(decimal, f"lacks the {COLUMN_ATTR!r} provenance block (zagg spec §4.6)")
        return None
    if block.get("spec") != COLUMN_SPEC:
        _skip(
            decimal,
            f"declares spec {block.get('spec')!r}; this reader implements {COLUMN_SPEC!r} only",
        )
        return None
    want_window = ALL_TOKEN if window is None else window
    if (
        block.get("node") != decimal
        or block.get("order") != shard_order
        or block.get("window") != want_window
    ):
        raise ValueError(
            f"column at node {decimal} declares node {block.get('node')!r}/order "
            f"{block.get('order')!r}/window {block.get('window')!r}, not this leaf's "
            f"{decimal!r}/{shard_order}/{want_window!r} — its groups would be read "
            f"under the wrong node (zagg spec §4.6)"
        )
    return {"node": decimal, "window": window, "role": "column", COLUMN_ATTR: block}


def open_column_order(
    store_root: str,
    manifest: dict,
    order: int,
    *,
    aoi=None,
    window: str | None = None,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    index_kind: str = "moc",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    store: Any = None,
    _envelope: dict | None = None,
    **store_kwargs: Any,
):
    """Open one column-carried resolution of a product as a Dataset, or ``None``.

    The §4.6 tier of the #37 per-level model: ``order`` is a CELL order in
    ``[shard_order, cell_order)`` — a declared leaf resolution
    (``pyramid_declaration(...)["leaf_cells"]``), a within-footprint ladder
    rung, or the node-order member ``order == shard_order`` (the leaves'
    whole-footprint aggregates: one cell per leaf) — and the returned
    dataset assembles that resolution group across every covered leaf's
    column, concatenated along ``cells`` in ascending packed-word order.
    The shape contract is :mod:`moczarr.level`'s: the same dims / ``morton``
    coordinate / attrs as :func:`moczarr.open.open_hive` returns for the
    native order, with dense fields dense and ragged digests as their
    encoded vlen variables (the ``ragged`` attrs block riding verbatim).

    Which resolutions a given column actually holds is the column's own
    ``groups`` map (the manifest MAY lag what the fleet wrote — §4.6), so
    membership is checked per column and a stamped column without this
    group contributes nothing, silently: the level then UNDER-COVERS those
    leaves' footprints, visible as absent spans of the ``morton`` domain
    (the same posture as a cascade's ``source_children.missing`` — absent
    is never evidence of empty). Leaves with no stamped column at all are
    skipped the same way (D4: absence of a stamp IS the answer; §4.6 makes
    torn-worker and never-declared indistinguishable here and neither an
    error). Per-object classification follows :func:`_column_entry`'s two
    severities, and each admitted column's ``zagg_column`` block rides
    verbatim in ``attrs["zagg_objects"]``.

    **The roster here is QUERY-SCOPED — a named exception.**
    :func:`moczarr.pyramid.open_overview_order` records ``zagg_objects``
    for every stamped object BEFORE any AOI filtering, so its roster (and
    its §4.3 classification) answers structural questions about the store;
    this surface does not. Because ``aoi`` cuts the CANDIDATE leaves (see
    below), both the roster and the §4.6 wrong-identity raise are functions
    of the query: an AOI that misses a leaf never reads that leaf's stamp,
    so a disjoint AOI yields ``zagg_objects == []`` on a store that carries
    columns, and a column declaring another node's identity raises only when
    the query reaches it (``aoi=None``, an AOI selecting it, or the
    unrestricted fallback probe below). That is deliberate — the exception
    buys the cost posture in the next paragraph — but it means structural
    questions ("does this store carry columns at all?", "is every column
    conformant?") belong to the unconditional surfaces:
    :func:`moczarr.store.walk_columns`,
    :func:`moczarr.column.read_column_record`, and
    :func:`moczarr.pyramid.read_pyramid`, never to an AOI-scoped level's
    roster. The three-way behavior is pinned in ``tests/test_level.py``.

    ``aoi`` restricts the CANDIDATE leaves arithmetically (root MOC ∩ AOI,
    the same shard-level cut :func:`moczarr.open.candidate_leaves` makes —
    columns are leaf-sibling artifacts, one per shard, so an unscoped probe
    would pay the leaf tier's own stamp GETs on every open: 2,918 on ATL03,
    against the 4^(s-k)-fold fewer ancestor nodes that let
    ``open_overview_order`` keep its candidates unscoped for free) and then
    rows exactly, per group. An AOI that excludes every column-carried cell
    returns the issue-#4 schema-correct empty dataset with a
    ``UserWarning`` — which names WHICH of the two empties it is: the AOI
    missing the column coverage, or the AOI's own stamped columns carrying
    no group ``r`` (under-coverage, the schema then coming from the
    fallback probe's out-of-AOI leaf); ``None``
    is returned — with a warning — only when NO stamped column anywhere
    carries this group (not yet written, or a declaration that never
    carried it), or when the root ``coverage.moc`` is unusable (candidates
    are named arithmetically, exactly as
    :func:`moczarr.pyramid.open_overview_order` refuses to walk). Unlike
    overviews, a grouped tree needs no refusal: a column is a leaf sibling,
    so ``path_grouping`` renders its path exactly as it renders the leaf's.

    ``window`` follows the leaf dialect (D23): required on a windowed
    (``morton-hive/2``) store — its columns are ``{window}.pyramid.zarr``
    per window, with NO all-time column (the ``all.pyramid.zarr`` stem is
    the unwindowed store's spelling, reached by ``window=None``) — refused
    on an unwindowed one, and the reserved token is refused at the shared
    ``validate_window`` seam. ``anonymous``/``store``/``_envelope``/
    ``store_kwargs`` follow :func:`moczarr.pyramid.open_overview_order`'s
    posture (one shared handle, one root-MOC read per product).
    """
    import xarray as xr
    from zarr.storage import ObjectStore

    from moczarr.coverage import as_moc_words
    from moczarr.open import _check_composition_fill
    from moczarr.store import _resolve_store

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    if fabricate_cell_ids not in ("auto", True, False):
        raise ValueError(
            f"fabricate_cell_ids={fabricate_cell_ids!r}: expected 'auto', True, or False"
        )
    if index_kind not in ("pandas", "moc"):
        raise ValueError(f"index_kind={index_kind!r}: expected 'pandas' or 'moc'")
    cell_order = int(manifest["cell_order"])
    shard_order = int(manifest["shard_order"])
    r = int(order)
    if not (shard_order <= r < cell_order):
        raise ValueError(
            f"order {r} is not a column-carried resolution of this store: §4.6 groups "
            f"live at cell orders shard_order <= r < cell_order "
            f"({shard_order} <= r < {cell_order}); the native order is open_hive's, "
            f"and coarser levels are ancestor artifacts (open_overview_order)"
        )
    grouping = manifest_path_grouping(manifest)
    windowed = manifest["spec"] == HIVE_SPEC_V2
    if window is not None:
        # The shared seam, ABOVE the windowed branch (the reserved token and
        # a label on an unwindowed store are refused unconditionally).
        validate_window(window, where=store_root)
    if windowed and window is None:
        raise ValueError(
            f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; its columns are "
            f"per-window ({{window}}.pyramid.zarr, D23 naming) — pass window=..."
        )
    if not windowed and window is not None:
        raise ValueError(
            f"window={window!r} on a {manifest['spec']} store: unwindowed stores "
            f"have no window leaves (schedule: none)"
        )

    obstore_store = _resolve_store(store_root, store, store_kwargs)
    envelope = _envelope
    if envelope is None:
        envelope = load_root_coverage(store_root, store=obstore_store)
    if envelope is None:
        warnings.warn(
            f"no usable root coverage.moc at {store_root}: candidate columns are "
            f"named arithmetically from the source domain, so column order {r} "
            f"cannot be enumerated and is omitted (regenerate the root coverage; "
            f"columns are derived artifacts a reader never requires — zagg spec §4.6)",
            UserWarning,
            stacklevel=2,
        )
        return None
    aoi_words = as_moc_words(aoi) if aoi is not None else None
    all_words = np.sort(ranges_words(envelope))
    if aoi_words is None:
        words = all_words
    else:
        # The shard-level cut candidate_leaves makes (open.py): the MOC
        # intersection keeps the finer element of each overlapping pair, so
        # coarsen back to shard order to name the columns.
        words = root_coverage_and(envelope, aoi_words)
        if words.size:
            from mortie import clip2order

            words = np.unique(clip2order(shard_order, words))
        words = np.sort(words)
    shards = [morton_decimal(int(w)) for w in words]
    rels = [column_path(dec, window, path_grouping=grouping) for dec in shards]
    metas = read_leaf_metas(store_root, rels, store=obstore_store, concurrency=concurrency)
    zarr_store = ObjectStore(obstore_store, read_only=True)

    def _open_group(rel: str):
        ds = xr.open_zarr(
            zarr_store,
            group=f"{rel}/{r}",
            consolidated=False,
            zarr_format=3,
            **(xr_kwargs or {}),
        )
        _check_composition_fill(ds, rel)
        coords = [name for name in ("morton", "cell_ids") if name in ds]
        return ds.set_coords(coords), coords

    opened, entries = [], []
    domain = None
    moc_dim = "cells"
    schema_rel = None
    admitted = 0  # candidates that classified, group membership aside
    for dec, rel, meta in zip(shards, rels, metas):
        stamp = _stamp_from_meta(meta)
        if stamp is None:
            continue  # no column, or unstamped debris (D4/§4.6) — never an error
        attrs = meta.get("attributes") if isinstance(meta, dict) else None
        entry = _column_entry(attrs or {}, dec, window, shard_order)
        if entry is None:
            continue  # malformed artifact, warned and dropped
        admitted += 1
        if str(r) not in (entry[COLUMN_ATTR].get("groups") or {}):
            continue  # this leaf's declaration carried no group r: under-coverage
        entries.append(entry)
        if schema_rel is None:
            schema_rel = rel
        shard_word = morton_word(dec)
        if index_kind == "moc":
            leaf_domain = MortonRanges.from_shards([shard_word], r)
            if aoi_words is not None:
                leaf_domain = leaf_domain.intersect(aoi_words)
                if leaf_domain.size == 0:
                    continue
        ds, coords = _open_group(rel)
        if index_kind == "moc":
            moc_dim = ds["morton"].dims[0] if "morton" in ds.coords else "cells"
            ds = ds.drop_vars(coords)
            if aoi_words is not None:
                full = MortonRanges.from_shards([shard_word], r)
                ds = ds.isel({moc_dim: full.rank(leaf_domain.fabricate())})
            domain = leaf_domain if domain is None else domain.union(leaf_domain)
        elif aoi_words is not None and "morton" in ds.coords:
            from moczarr.coverage import aoi_mask

            keep = aoi_mask(np.asarray(ds["morton"].values, dtype=np.uint64), aoi_words)
            if not keep.any():
                continue
            ds = ds.isel({ds["morton"].dims[0]: keep})
        opened.append(ds)
    if schema_rel is None and aoi_words is not None:
        # The AOI cut the candidate list itself, so "no admitted column" is
        # ambiguous between not-written and AOI-excluded. Resolve it on the
        # exceptional path only: probe the UNRESTRICTED roster for one
        # stamped column carrying this group — the issue-#4 empty return
        # needs a schema object, and its absence store-wide is the honest
        # None (open_overview_order's unscoped candidates give it the same
        # split for free at 4^(s-k)-fold lower probe cost).
        seen = set(shards)
        rest = [dec for dec in (morton_decimal(int(w)) for w in all_words) if dec not in seen]
        rest_rels = [column_path(dec, window, path_grouping=grouping) for dec in rest]
        rest_metas = read_leaf_metas(
            store_root, rest_rels, store=obstore_store, concurrency=concurrency
        )
        for dec, rel, meta in zip(rest, rest_rels, rest_metas):
            if _stamp_from_meta(meta) is None:
                continue
            attrs = meta.get("attributes") if isinstance(meta, dict) else None
            entry = _column_entry(attrs or {}, dec, window, shard_order)
            if entry is not None and str(r) in (entry[COLUMN_ATTR].get("groups") or {}):
                schema_rel = rel
                break
    if schema_rel is None:
        scope = f" window {window!r}" if windowed else ""
        warnings.warn(
            f"no stamped column at {store_root}{scope} carries resolution group {r}; "
            f"level omitted (not yet written, or the writing declarations carried "
            f"no group there — columns are derived artifacts a reader never "
            f"requires, zagg spec §4.6)",
            UserWarning,
            stacklevel=2,
        )
        return None
    if not opened:
        # Two distinguishable states land here, and conflating them misreads
        # the store: the AOI really missed the column coverage, OR it
        # selected stamped columns that simply do not carry group r (the
        # §4.6 under-coverage the docstring describes — the schema then came
        # from the fallback probe's out-of-AOI leaf). `entries` is the
        # admitted-AND-carrying list, `admitted` the admitted count.
        if entries or not admitted:
            reason = (
                f"the given AOI intersects no column coverage at {store_root} (column order {r})"
            )
        else:
            reason = (
                f"the given AOI selected {admitted} stamped column(s) at {store_root}, "
                f"none of which carries resolution group {r}: the level UNDER-COVERS "
                f"those leaves rather than the AOI missing coverage (the declaration "
                f"MAY lag the fleet — zagg spec §4.6)"
            )
        warnings.warn(
            f"{reason}; returning a schema-correct empty dataset (0 cells)",
            UserWarning,
            stacklevel=2,
        )
        ds, coords = _open_group(schema_rel)
        empty_dim = ds["morton"].dims[0] if "morton" in ds.coords else "cells"
        if index_kind == "moc":
            moc_dim = empty_dim
            ds = ds.drop_vars(coords)
            domain = MortonRanges(np.empty((0, 2), dtype=np.uint64), r)
        opened.append(ds.isel({empty_dim: slice(0)}))
    dim = opened[0]["morton"].dims[0] if "morton" in opened[0].coords else "cells"
    if index_kind == "moc":
        dim = moc_dim
    result = xr.concat(opened, dim=dim) if len(opened) > 1 else opened[0]
    if index_kind == "moc":
        from moczarr.moc_index import MortonMocIndex

        assert domain is not None  # a column set it, or the empty path did
        index = MortonMocIndex(domain, dim=dim, name="morton")
        result = result.assign_coords(xr.Coordinates.from_xindex(index))
    if "morton" in result.coords and (
        fabricate_cell_ids is True
        or (fabricate_cell_ids == "auto" and "cell_ids" not in result.coords)
    ):
        from moczarr.fabricate import fabricate_cell_ids as _fabricate

        ids = _fabricate(
            np.asarray(result["morton"].values, dtype=np.uint64),
            level=r,
            _stacklevel=4,
        )
        result = result.assign_coords(cell_ids=(result["morton"].dims, ids))
    result = result[sorted(result.data_vars)]
    from moczarr.pyramid import OBJECTS_ATTR

    result.attrs["morton_hive"] = {
        "spec": manifest["spec"],
        "cell_order": r,
        "shard_order": shard_order,
        "dataset": manifest.get("dataset"),
    }
    result.attrs[OBJECTS_ATTR] = entries
    return result


def _resolve_target(
    store_root: str,
    product: str | None,
    manifest: dict | None,
    store: Any,
    store_kwargs: dict[str, Any],
) -> tuple[str, Any, dict]:
    """``(store_root, obstore handle, manifest)`` for one product subtree.

    The shared front door of the resolution-addressed entry points
    (:func:`open_level`, :func:`open_pyramid`): D19 ``product=`` re-rooting,
    one handle (threaded or constructed), and one manifest (threaded and
    re-parsed, or read through the handle) — with the pointed multi-product
    error when a root manifest is missing because the root is a directory
    of stores rather than a store.
    """
    from moczarr.store import _resolve_store, read_manifest

    if product is not None:
        from moczarr.products import validate_product_name

        validate_product_name(product)
        store_root = f"{store_root.rstrip('/')}/{product}"
    handle = _resolve_store(store_root, store, store_kwargs)
    if manifest is None:
        manifest = read_manifest(store_root, store=handle)
        if manifest is None:
            if product is None:
                from moczarr.products import list_products

                names = [p["name"] for p in list_products(store_root, store=handle)]
                if names:
                    raise ValueError(
                        f"{store_root} is a multi-product store root (products: {names}); "
                        f"pass product=... to open one (D19, mortie spec §6.5)"
                    )
            raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    else:
        from moczarr.convention import parse_manifest

        manifest = parse_manifest(manifest)
    return store_root, handle, manifest


def open_level(
    store_root: str,
    cell_order: int,
    *,
    product: str | None = None,
    manifest: dict | None = None,
    aoi=None,
    window: str | None = None,
    all_time: bool = False,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    index_kind: str = "moc",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    store: Any = None,
    _envelope: dict | None = None,
    **store_kwargs: Any,
):
    """Open ONE resolution level of a pyramid store as one Dataset, or ``None``.

    The #37 entry point: ``cell_order`` addresses a level of
    :func:`pyramid_levels`' table — resolution is the reader-facing axis,
    and which artifact kind materializes it is this function's dispatch,
    not the caller's:

    - the **native** order opens through :func:`moczarr.open.open_hive`
      (the same lazy Dataset, plus the per-object roster under
      ``attrs["zagg_objects"]``);
    - an **overview** order opens through
      :func:`moczarr.pyramid.open_overview_order` (ancestor artifacts,
      ``/1`` constant-depth and ``/2`` ladder rungs alike);
    - a **column** order opens through :func:`open_column_order` (the §4.6
      leaf tier of a ``/2`` declaration).

    Every arm returns the ONE shape this module's docstring specifies, and
    every arm inherits its opener's degrade postures verbatim: ``None``
    with a warning when a declared level has no stamped artifact (or the
    root MOC is unusable), the issue-#4 schema-correct empty dataset under
    an AOI that misses the coverage, and the shared ``window=`` seam. An
    undeclared ``cell_order`` raises, listing the store's levels — the
    §4.6 node-order member (``cell_order == shard_order``) is a recorded
    partial tier rather than a level, reachable explicitly through
    :func:`open_column_order`.

    On top of the arm's own attrs the result carries :data:`LEVEL_ATTR`
    (``zagg_level``) — the level's record from the table plus the
    declaration-side context a per-level consumer needs without re-reading
    the manifest::

        {"cell_order": 5, "order": 4, "artifact": "column",
         "spec": "zagg-pyramid/2",
         "fields": {...},          # the §4.5 all-fields map, classes included
         "fold_source": "cascade"}

    ``fields`` is the DECLARED all-fields view (a ``none``-class entry
    records an absence — the zero-open §4.4 answer); per-artifact truth
    (what a given object actually folded, its regime and generation, any
    stamped ``demotions``) stays in the ``zagg_objects`` roster, per
    object, never summarized. ``spec``/``fields``/``fold_source`` are
    ``None`` on a declared-off store, whose one level is the native order.

    ``all_time`` opens a windowed store's §4.5 CROSS-WINDOW folds at this
    level (issue #31, the recorded option-(1) posture): overview levels
    only — a windowed store has no all-time source leaf (the fold is *of*
    the windows) and no all-time column (§4.6: the ``all.pyramid.zarr``
    stem is the unwindowed store's spelling) — so naming a source or
    column level under ``all_time=True`` raises, and the arm's own seams
    refuse it on an unwindowed store (whose artifacts ARE the all-time
    folds, reached with ``window=None``) and beside a ``window=``. What the
    *declaration* says is deliberately NOT a seam here: a store whose
    ``pyramid.overview.all_time`` is unset still reads whatever ``all.zarr``
    folds are stamped (declaring is free, sweeping is the operational
    decision — zagg#381 point (11)), degrading to the ordinary warned
    ``None`` when none are. Only :func:`open_pyramid` gates on the
    declaration, where the alternative is a childless tree of warnings.

    ``product`` re-roots on a D19 multi-product subtree; ``manifest``
    threads an already-read manifest (of the subtree actually opened) and
    ``store`` one already-constructed obstore handle — both reach EVERY
    arm, the native one included (issue #5: one store construction and one
    manifest GET per open, whichever artifact kind owns the level, so
    threading a handle across a product's levels behaves uniformly); the
    private ``_envelope`` likewise shares one already-read root MOC across
    a product's non-source levels (how :func:`open_pyramid` calls this —
    the sidecar tier must not grow with the number of levels);
    everything else follows :func:`moczarr.open.open_hive`'s posture.
    Deliberately no ``decode=``: new pyramid surfaces are native moczarr
    only (the englacial/zagg#550 ruling) — the returned Dataset is plain
    xarray plus the core lazy index.
    """
    from moczarr.pyramid import OBJECTS_ATTR, open_overview_order

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    store_root, handle, manifest = _resolve_target(
        store_root, product, manifest, store, store_kwargs
    )
    levels = pyramid_levels(manifest)
    decl = pyramid_declaration(manifest)
    record = levels.get(int(cell_order))
    if record is None:
        from moczarr.pyramid import PYRAMID_SPEC_V2

        hint = (
            " (the §4.6 node-order partial tier is not a level; open it explicitly "
            "with open_column_order)"
            if decl is not None
            and decl["spec"] == PYRAMID_SPEC_V2
            and int(cell_order) == int(manifest["shard_order"])
            else ""
        )
        raise ValueError(
            f"cell order {cell_order} is not a level of {store_root}: this store's "
            f"levels are {list(levels)} (finest first; see pyramid_levels){hint}"
        )
    if all_time and record["artifact"] != "overview":
        raise ValueError(
            f"all_time=True reads the §4.5 cross-window FOLDS, which exist only at "
            f"overview levels: level {cell_order} is {record['artifact']!r} (a windowed "
            f"store has no all-time source leaf and no all-time column — issue #31, "
            f"zagg spec §4.5/§4.6)"
        )
    if record["artifact"] == "source":
        from moczarr.open import open_hive

        objects: list[dict] = []
        ds = open_hive(
            store_root,
            aoi=aoi,
            window=window,
            fabricate_cell_ids=fabricate_cell_ids,
            index_kind=index_kind,
            concurrency=concurrency,
            xr_kwargs=xr_kwargs,
            # The same handle and the same manifest the other two arms get
            # (issue #5). Without them this arm alone re-resolved a store
            # from store_root and re-GET morton_hive.json, so a caller
            # threading ONE handle across a product's levels had N-1 shared
            # reads and one unshared one — and a handle carrying config that
            # store_kwargs does not (or rooted where store_root is not)
            # worked on every level EXCEPT the one every store has.
            store=handle,
            manifest=manifest,
            _objects_out=objects,
            **store_kwargs,
        )
        ds.attrs[OBJECTS_ATTR] = objects
    else:
        envelope = _envelope
        if envelope is None:
            envelope = load_root_coverage(store_root, store=handle)
        common = dict(
            aoi=aoi,
            window=window,
            fabricate_cell_ids=fabricate_cell_ids,
            index_kind=index_kind,
            concurrency=concurrency,
            xr_kwargs=xr_kwargs,
            store=handle,
            _envelope=envelope,
        )
        if record["artifact"] == "column":
            ds = open_column_order(store_root, manifest, record["cell_order"], **common)
        else:
            # The declared resolution is threaded, never re-derived: for /1
            # it EQUALS the constant-depth default, for /2 it is the ladder
            # entry's cells member (the recorded list is the contract).
            ds = open_overview_order(
                store_root,
                manifest,
                record["order"],
                cell_order=record["cell_order"],
                all_time=all_time,
                **common,
            )
    if ds is None:
        return None
    ds.attrs[LEVEL_ATTR] = {
        **record,
        "spec": decl["spec"] if decl else None,
        "fields": decl["fields"] if decl else None,
        "fold_source": decl["fold_source"] if decl else None,
    }
    return ds


def open_pyramid(
    store_root: str,
    *,
    product: str | None = None,
    manifest: dict | None = None,
    levels=None,
    aoi=None,
    window: str | None = None,
    all_time: bool = False,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    index_kind: str = "moc",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    store: Any = None,
    **store_kwargs: Any,
):
    """Open a pyramid store's whole resolution ladder as one ``xarray.DataTree``.

    The espg/moczarr#36b multi-order assembly over :func:`pyramid_levels`:
    one child group per addressable resolution, finest first, each named by
    the integer cell order it stores and holding exactly the Dataset
    :func:`open_level` returns for that resolution (the #37 shape — the
    ``cells`` axis, the lazy ``morton`` coordinate, ragged digests encoded,
    ``zagg_level``/``zagg_objects`` attrs) — whichever artifact kind
    materializes it (source leaves, §4.6 column groups, §4.1/§4.4 overview
    artifacts), on either declaration grammar (``/1`` constant-depth and
    the ``/2`` fixed ladder alike). The tree layer adds no reads of its
    own beyond one manifest GET and one root-MOC read shared across the
    levels (issue #5), and "lazy" stays each level's own laziness.

    The **root node is empty** — no variables, no coordinates — and carries
    the declaration:

    - ``morton_hive`` — the manifest summary (``spec``, the native
      ``cell_order``, ``shard_order``, ``dataset``), plus ``semantic_hash``
      when the manifest records one;
    - ``zagg_pyramid`` — the normalized :func:`moczarr.pyramid.
      pyramid_declaration` record, decoded from the NORMATIVE ``pyramid``
      block (absent on a declared-off store, whose tree is the valid
      one-level degenerate form: the native order and nothing else);
    - ``multiscales`` — the manifest's §4.9 ``zagg-multiscales/1``
      discovery mirror **verbatim**, when the manifest carries one. It is
      surfaced as recorded, never re-derived and never consulted: the
      levels come from the ``pyramid`` block, which wins any disagreement
      by §4.9's own precedence rule.

    A declared level with no stamped artifact is OMITTED from the tree with
    its opener's warning (declared-but-unmaterialized is a legal recorded
    state — declaring is free, sweeping is the operational decision), and
    an ``aoi`` scopes ROWS per level, never the tree's shape: an
    out-of-coverage AOI empties each materialized level schema-correct
    (issue #4). The seamless mixed-order composite stays a *computed* view,
    never a node, and zoom-out ergonomics (a default coarse level for a
    first render) stay espg/moczarr#21's.

    ``levels`` restricts the assembly to a subset of the declared cell
    orders (a bounded-cost open on a wide ladder — a ``/2`` store declares
    every order down to 0); an order outside :func:`pyramid_levels`' table
    raises, listing the store's levels. ``product`` re-roots on a D19
    multi-product subtree, ``manifest``/``store`` thread an already-read
    manifest and an already-constructed handle, and ``window`` follows each
    arm's D23 seam (required on a windowed store, refused on an unwindowed
    one, the reserved all-time token refused everywhere). Deliberately no
    ``decode=``: pyramid surfaces are native moczarr only (the
    englacial/zagg#550 ruling).

    ``all_time`` is the windowed store's cross-window surface (issue #31,
    the recorded option-(1) posture): the tree's groups are the §4.5
    all-time FOLD levels alone — overviews-only, honest about what is
    materialized. The source level and any §4.6 column levels are ABSENT
    (a windowed store has no all-time source leaf and no all-time column),
    unless ``levels=`` names one explicitly, which raises instead of
    silently dropping the request. Refused beside ``window=`` (the fold
    sums every window), on an unwindowed store (whose artifacts ARE its
    all-time folds — open them with ``window=None``), and — THIS opener
    alone, see :func:`open_level` — on a declaration without ``all_time``
    folds, because the whole-ladder alternative is a childless tree of
    warnings rather than a pointed answer.
    """
    import xarray as xr

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    store_root, handle, manifest = _resolve_target(
        store_root, product, manifest, store, store_kwargs
    )
    table = pyramid_levels(manifest)
    decl = pyramid_declaration(manifest)
    if levels is not None:
        wanted = [int(r) for r in levels]
        if not wanted:
            raise ValueError(
                f"levels=[] selects nothing at {store_root} — omit levels= to open "
                f"every declared level"
            )
        bad = sorted(set(wanted) - set(table))
        if bad:
            raise ValueError(
                f"cell orders {bad} are not levels of {store_root}: this store's "
                f"levels are {list(table)} (finest first; see pyramid_levels)"
            )
        table = {r: table[r] for r in table if r in set(wanted)}
    if all_time:
        # issue #31, the recorded option-(1) posture: the all-time view is
        # overviews-only. The window/unwindowed seams are re-checked by the
        # per-level arm; the declaration gate lives here because an
        # undeclared all_time would otherwise degrade to a childless tree
        # (every level None) instead of the pointed answer.
        if manifest["spec"] != HIVE_SPEC_V2:
            raise ValueError(
                f"all_time=True on a {manifest['spec']} store: an unwindowed store's "
                f"ancestor artifacts ARE its all-time folds — open them with window=None"
            )
        if window is not None:
            raise ValueError(
                f"window={window!r} with all_time=True: the §4.5 all-time fold sums "
                f"EVERY window, so the two selections are mutually exclusive"
            )
        if decl is None or not decl["all_time"]:
            raise ValueError(
                f"{store_root} declares no all-time folds "
                f"(pyramid.overview.all_time is not set), so there is nothing to open "
                f"(zagg spec §4.5; declaring is the writer's choice)"
            )
        if levels is None:
            # The source level (and any §4.6 column level) has no all-time
            # artifact by construction — absent from the tree, not empty.
            table = {r: rec for r, rec in table.items() if rec["artifact"] == "overview"}
        # An explicit levels= naming a non-overview level flows to
        # open_level's own all_time raise below — never a silent drop.
    envelope = None
    if any(rec["artifact"] != "source" for rec in table.values()):
        # ONE root-MOC read for the whole ladder (issue #5): the non-source
        # arms name their candidates from it. None means "not usable" and the
        # per-level opens then warn exactly as they do standalone.
        envelope = load_root_coverage(store_root, store=handle)
    nodes: dict[str, Any] = {}
    for r in table:
        ds = open_level(
            store_root,
            r,
            manifest=manifest,
            aoi=aoi,
            window=window,
            all_time=all_time,
            fabricate_cell_ids=fabricate_cell_ids,
            index_kind=index_kind,
            concurrency=concurrency,
            xr_kwargs=xr_kwargs,
            store=handle,
            _envelope=envelope,
            **store_kwargs,
        )
        if ds is None:
            continue  # unmaterialized level — the arm warned; the group is omitted
        nodes[str(r)] = ds
    root_attrs: dict[str, Any] = {
        "morton_hive": {
            "spec": manifest["spec"],
            "cell_order": int(manifest["cell_order"]),
            "shard_order": int(manifest["shard_order"]),
            "dataset": manifest.get("dataset"),
        }
    }
    if manifest.get("semantic_hash"):
        root_attrs["semantic_hash"] = manifest["semantic_hash"]
    if decl is not None:
        root_attrs["zagg_pyramid"] = decl
    if manifest.get("multiscales") is not None:
        root_attrs["multiscales"] = manifest["multiscales"]
    root = xr.Dataset(attrs=root_attrs)
    return xr.DataTree.from_dict({"/": root, **nodes})


def level_demotions(ds) -> list[dict]:
    """Every stamped ``demotions`` record of one level, node-attributed.

    **Pre-spec, forward-tolerant, surfaced verbatim.** The ``demotions``
    key is NOT in the normative store contract: zagg's byte-level grammar
    is ``docs/specification.md`` plus the conformance fixtures (zagg#340),
    and this key lives only in the open, ``waiting`` PR englacial/zagg#557
    (the packed-class guard rail, zagg#518 — one record per
    ``(field, reason)``, e.g. ``{"field": "composition", "class":
    "packed", "reason": "divisor-missing", ...}``). So this function makes
    NO claim about the record shape and validates nothing: it reads the
    key where the PR says the sweep writes it — the affected overview
    artifact's ``zagg_overview`` attrs, which ride VERBATIM in this model's
    ``zagg_objects`` roster — flattens the records across the level's
    artifacts, and extends each with the carrying object's ``node`` and
    ``window``. Unknown record keys pass through untouched, and absence
    everywhere is the clean answer ``[]`` (a pre-#518 writer's attrs are
    byte-identical to a clean store's, so absence is never evidence the
    rail stayed quiet). If #557's shape moves before it lands, this
    docstring and ``tests/test_level.py``'s hand-written record are the
    two places to update; when it lands with a spec section and a fixture,
    the test re-pins on writer bytes.

    Read from ``zagg_overview`` ONLY. §4.6 leaf columns are written by the
    leaf worker, which runs no fold, and #557's instrumented sites are all
    sweep-side, so a ``zagg_column`` block has no producer for this key —
    reading one would be speculation dressed as a surface.

    Takes an :func:`open_level` /
    :func:`moczarr.pyramid.open_overview_order` /
    :func:`open_column_order` result — or a DataTree node wrapping one; a
    column level's answer is ``[]`` by construction.
    """
    node = ds.ds if hasattr(ds, "ds") else ds
    from moczarr.pyramid import OBJECTS_ATTR, OVERVIEW_ATTR

    out = []
    for entry in node.attrs.get(OBJECTS_ATTR) or []:
        block = entry.get(OVERVIEW_ATTR) or {}
        for record in block.get("demotions") or []:
            out.append({"node": entry.get("node"), "window": entry.get("window"), **record})
    return out


__all__ = [
    "LEVEL_ATTR",
    "level_demotions",
    "open_column_order",
    "open_level",
    "open_pyramid",
    "pyramid_levels",
]

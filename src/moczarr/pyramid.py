"""Resolution (pyramid-order) nodes: the manifest-declared overview axis.

Phase 8b of the DataTree reader model (issue #15). When a product's manifest
declares sweep-generated overviews (the zagg spec §4 ``pyramid`` block), the
resolution axis becomes a second node level under the product —
``{product}/{order}``, nodes named by the integer **cell order they store**
(zagg#201's ratified layout) — and this module supplies the order-node
opener plus the selection helpers :func:`moczarr.open.open_store` builds
that level from.

Node **discovery is manifest declaration** (zagg spec §4.5, resolving the
question the 8a design section left open): the reader binds
``pyramid.overview.orders`` — ``[]`` (or a pre-pyramid manifest) means the
family is declared off and today's flat product node stands — and nothing
else. Never a listing walk, never a MOC extension. The root ``coverage.moc``
stays the *source* domain (zagg#201 ruling (5)): the overview objects at a
declared ancestor order are named arithmetically by coarsening that domain's
shards to the ancestor prefix, so enumerating an order node costs the same
metadata tier as a leaf open (stamp GETs; zero chunk reads on the moc
default).

``role`` is per **object**, never per node (ruling (5) again): an order node
is in general a *mixture* of ``role: overview`` zarrs and coarse *source*
zarrs written natively at that order (D24 regional heterogeneity), so every
object's classification is checked on open and surfaced per object under
``attrs["zagg_objects"]`` — this reader never synthesizes a node-level role.
For the same D24 reason "the source node" is not a well-formed request:
source-order uniqueness is per (shard, window), not per product, and the
selection helpers below (:func:`source_orders`, :func:`overview_orders`,
:func:`finest_source_at`) range over the **set** of orders carrying each
role, keying on the per-object entries and never on node names.

Per-node variable sets differ in **both** directions and each node's schema
comes from its own objects: ``none``-class fields exist only at native
resolution (option A — absence at an overview order is an answer, not an
error), and declared derived summaries (option B / Phase F, zagg-side) may
exist at *no* source order — both tolerated here by construction. The
seamless mixed-order composite stays a *computed* view (truncation-join
arithmetic), never a tree node.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from moczarr.convention import (
    HIVE_SPEC_V2,
    decimal_base,
    manifest_path_grouping,
    morton_decimal,
    morton_word,
    validate_label,
)
from moczarr.coverage import ranges_words
from moczarr.ranges import MortonRanges
from moczarr.store import load_root_coverage, read_leaf_metas

#: Version string of the manifest ``pyramid`` block this reader binds (§4.5).
PYRAMID_SPEC = "zagg-pyramid/1"
#: Version string of the per-overview provenance attrs payload (§4.3).
OVERVIEW_SPEC = "zagg-overview/1"
#: Root-group attrs key classifying a zarr (D11: per object, never inferred
#: from tree position; source zarrs carry no role — absence means source).
ROLE_ATTR = "role"
#: Root-group attrs key carrying the overview provenance block (§4.3).
OVERVIEW_ATTR = "zagg_overview"
#: Dataset-attrs key carrying one order node's per-object entries.
OBJECTS_ATTR = "zagg_objects"
#: Reserved window token naming the unwindowed / all-time fold (D23).
ALL_TOKEN = "all"


def overview_declaration(manifest: dict) -> dict | None:
    """The manifest's overview declaration, or ``None`` when declared off.

    The §4.5 binding rule: branch on ``pyramid.overview.orders`` **first** —
    an empty ``orders``, a missing ``overview`` mapping, or no ``pyramid``
    block at all (pre-pyramid manifests, and the legacy ``{"orders": []}``
    placeholder shape older fixtures carry) all mean *no overview family
    exists* and no other key of the block may be assumed. When ``orders`` is
    non-empty the block's ``spec`` is strict-checked (fail loudly on an
    unknown revision — the conformance rule) and ``spacing`` / ``all_time`` /
    ``fields`` MUST all be present; additional keys (``summarize``, the
    sweep's ``materialized`` actuals, per-field ``nan_policy``…) are
    tolerated per the spec.
    """
    block = manifest.get("pyramid")
    if not isinstance(block, dict):
        return None
    overview = block.get("overview")
    if not isinstance(overview, dict) or not overview.get("orders"):
        return None
    spec = block.get("spec")
    if spec != PYRAMID_SPEC:
        raise ValueError(
            f"manifest pyramid block declares spec {spec!r}; this reader implements "
            f"{PYRAMID_SPEC!r} only (zagg spec §4.5 — strict-check, fail loudly)"
        )
    missing = [key for key in ("spacing", "all_time", "fields") if key not in overview]
    if missing:
        raise ValueError(
            f"manifest pyramid.overview declares orders {overview['orders']} but lacks "
            f"{missing}: with a non-empty orders list, spacing/all_time/fields MUST all "
            f"be present (zagg spec §4.5)"
        )
    shard_order = int(manifest["shard_order"])
    orders = [int(k) for k in overview["orders"]]
    bad = [k for k in orders if not (0 <= k < shard_order)]
    if bad:
        raise ValueError(
            f"manifest pyramid.overview.orders {bad} are not ancestor orders of "
            f"shard_order {shard_order} (must satisfy 0 <= k < shard_order)"
        )
    return overview


def overview_cell_orders(manifest: dict) -> dict[int, int]:
    """``{declared ancestor order: overview cell order}``, finest first.

    The §4.4 constant-depth rule: an overview at ancestor order ``k`` of a
    product with shard order ``s`` and cell order ``c`` holds cells at
    ``c - (s - k)`` — so these cell orders are the ``{product}/{order}``
    node names the declaration implies. Empty when the pyramid is declared
    off (the flat-product case).
    """
    decl = overview_declaration(manifest)
    if decl is None:
        return {}
    cell_order = int(manifest["cell_order"])
    shard_order = int(manifest["shard_order"])
    orders = sorted({int(k) for k in decl["orders"]}, reverse=True)
    return {k: cell_order - (shard_order - k) for k in orders}


def _node_rel(decimal: str) -> str:
    """An ancestor node decimal's relative path — one component per digit.

    The writer's convention verbatim (``zagg.sweep._node_rel``, which takes no
    grouping): ``"-311"`` -> ``"-3/1/1"``. Only called on a ``path_grouping:
    1`` store, where it is also the leaf tree's own node path; grouped stores
    are refused in :func:`open_overview_order` rather than guessed at.
    """
    base = decimal_base(decimal)
    return "/".join([base, *decimal[len(base) :]])


def _object_entry(attrs: dict, decimal: str, window: str | None, target_order: int) -> dict:
    """One stamped object's per-object entry; validates the §4.3 contract.

    ``role`` absence means source; ``role: "overview"`` requires the
    ``zagg_overview`` provenance block (present exactly then), whose ``spec``
    is strict-checked and whose ``cell_order`` must match this node's order —
    an off-order overview under a node would mis-rank every row after it.
    Any other ``role`` value breaks the closed two-value vocabulary and
    raises rather than guessing.
    """
    role = attrs.get(ROLE_ATTR)
    entry: dict[str, Any] = {"node": decimal, "window": window, "role": role or "source"}
    if role is None:
        return entry
    if role != "overview":
        raise ValueError(
            f"object at node {decimal} declares role {role!r}: the role vocabulary is "
            f"closed two-value — 'overview', or no role key at all (source); zagg spec §4.3"
        )
    block = attrs.get(OVERVIEW_ATTR)
    if not isinstance(block, dict):
        raise ValueError(
            f"role: overview object at node {decimal} lacks the {OVERVIEW_ATTR!r} "
            f"provenance block (present exactly when role is 'overview'; zagg spec §4.3)"
        )
    if block.get("spec") != OVERVIEW_SPEC:
        raise ValueError(
            f"overview at node {decimal} declares spec {block.get('spec')!r}; this reader "
            f"implements {OVERVIEW_SPEC!r} only (strict-check, fail loudly)"
        )
    if int(block["cell_order"]) != target_order:
        raise ValueError(
            f"overview at node {decimal} stores cells at order {block['cell_order']}, "
            f"not this node's order {target_order} — off-order objects would mis-rank rows"
        )
    entry[OVERVIEW_ATTR] = block
    return entry


def open_overview_order(
    store_root: str,
    manifest: dict,
    order: int,
    *,
    aoi=None,
    window: str | None = None,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    decode: bool = False,
    index_kind: str = "moc",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    store: Any = None,
    **store_kwargs: Any,
):
    """Open one declared overview order of a product as a Dataset, or ``None``.

    ``order`` is the declared **ancestor** order ``k`` from
    ``pyramid.overview.orders``; the returned dataset holds cells at the
    §4.4 cell order ``c - (s - k)`` (the node's name in the tree). Candidate
    objects are named arithmetically — the root MOC's source shards coarsened
    to the order-``k`` prefix, one ``{window}.zarr`` (or ``all.zarr``) per
    ancestor node — and each is admitted by its commit stamp (unstamped
    debris skipped, D4) and classified by its own ``role`` attrs, surfaced
    per object under ``attrs[OBJECTS_ATTR]``.

    Returns ``None`` when the order has no stamped object at all (declared
    but not yet swept — the caller omits the node), when the root MOC is
    unusable, or when the store declares ``path_grouping > 1`` (the grouped
    ancestor-path convention is unsettled writer-side). All three degrade by
    omission with a warning, never to a listing walk or a guessed path:
    overviews are regenerable caches and never load-bearing (§4.1), so the
    source node stands in every case. An ``aoi`` that excludes every stamped
    object returns the
    issue-#4 schema-correct empty dataset with a ``UserWarning``, exactly
    like :func:`moczarr.open.open_hive`.

    ``window`` must be a declared label: the reserved all-time token
    (``"all"``, §4.2) is **refused** on a windowed product. Its ``all.zarr``
    folds exist on disk (``pyramid.overview.all_time``, §4.5) but have no
    counterpart on the source axis, so surfacing them alone would report a
    0-cell source node beside overview nodes summing every window — the
    opt-in surface is deferred rather than half-built.

    ``anonymous`` skips request signing for public buckets, and remaining
    ``store_kwargs`` reach ``open_object_store`` — the same posture (and
    spelling) as :func:`moczarr.open.open_hive`, so the remote read path is
    declared in the signature rather than smuggled through ``store_kwargs``.
    """
    import xarray as xr
    from zarr.storage import ObjectStore

    from moczarr.open import _aoi_words, _check_composition_fill
    from moczarr.store import _resolve_store, _stamp_from_meta

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
    k = int(order)
    if not (0 <= k < shard_order):
        raise ValueError(f"order {k} is not an ancestor order of shard_order {shard_order}")
    target_order = cell_order - (shard_order - k)
    grouping = manifest_path_grouping(manifest)
    windowed = manifest["spec"] == HIVE_SPEC_V2
    if windowed:
        if window is None:
            raise ValueError(
                f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; its overview "
                f"orders are per-window (D23 naming) — pass window=..."
            )
        validate_label(window)
        if window == ALL_TOKEN:
            # `all` passes validate_label, so without this it would happily
            # open the all-time folds while open_hive found no `{shard}_all`
            # leaf and emptied the source node — one tree whose source order
            # reports 0 cells beside overview orders summing EVERY window.
            # Refusing is spec-aligned (§4.2 excludes `all` from the window
            # grammar forever), and leaves the surface question open.
            raise ValueError(
                f"window={ALL_TOKEN!r} at {store_root}: {ALL_TOKEN!r} is the reserved "
                f"all-time token (zagg spec §4.2 — excluded from the window grammar "
                f"forever), not a window label. The all-time overview folds "
                f"(pyramid.overview.all_time, §4.5) exist on disk but are not yet a "
                f"reader surface: the source axis has no all-time leaf to pair them "
                f"with, so opening them alone would mislead. Pass a declared window "
                f"label (espg/moczarr#15 tracks the opt-in surface)"
            )
        basename = f"{window}.zarr"
    else:
        basename = f"{ALL_TOKEN}.zarr"
    if grouping != 1:
        # An ancestor order that is not a multiple of path_grouping has NO
        # directory in a grouped tree (its component would be a truncated
        # group — `4/33` where the tree's component is `331`), and zagg's
        # sweep writes ancestor nodes per digit regardless of grouping
        # (`zagg.sweep._node_rel` takes no grouping argument). Any guess here
        # names a path that is neither the writer's nor a node of the leaf
        # tree, so the order is omitted loudly and the source node stands —
        # overviews are §4.1 regenerable caches a reader MUST NOT require.
        warnings.warn(
            f"{store_root} declares path_grouping {grouping}: the grouped-tree path of "
            f"an overview ancestor node is not settled (zagg's sweep writes ancestor "
            f"nodes one component per digit regardless of grouping), so declared order "
            f"{k} is omitted rather than guessed at; the source node is unaffected "
            f"(overviews are never load-bearing — zagg spec §4.1)",
            UserWarning,
            stacklevel=2,
        )
        return None

    obstore_store = _resolve_store(store_root, store, store_kwargs)
    envelope = load_root_coverage(store_root, store=obstore_store)
    if envelope is None:
        warnings.warn(
            f"no usable root coverage.moc at {store_root}: overview order nodes are "
            f"named by coarsening the source domain, so declared order {k} cannot be "
            f"enumerated and is omitted (regenerate the root coverage; overviews are "
            f"never load-bearing)",
            UserWarning,
            stacklevel=2,
        )
        return None
    aoi_words = _aoi_words(aoi) if aoi is not None else None
    # Candidates are ALL covered ancestors — the AOI scopes rows per object
    # below (and empties to the issue-#4 posture), never the candidate set:
    # tree shape must not depend on the AOI, and the schema for the empty
    # return needs a stamped object to read. Ancestor nodes are 4**(s-k)-fold
    # fewer than shards, so the unscoped stamp GETs stay cheap.
    shard_words = ranges_words(envelope)
    # Ordered by PACKED WORD, never by decimal string: a leading "-" (southern
    # base cell) sorts before every digit as a string but AFTER every northern
    # base cell as a word, so on a store spanning both hemispheres a decimal
    # sort lays rows down in an order the moc coordinate — accumulated in
    # `domain`, word-ascending — disagrees with, and §4.4 makes the coordinate
    # the truth for row identity. This is the invariant `_candidate_leaves`
    # keeps by sorting words before naming leaves (open.py).
    ancestors = sorted(
        {
            dec[: len(decimal_base(dec)) + k]
            for dec in (morton_decimal(int(w)) for w in shard_words)
        },
        key=morton_word,
    )
    rels = [f"{_node_rel(dec)}/{basename}" for dec in ancestors]
    metas = read_leaf_metas(store_root, rels, store=obstore_store, concurrency=concurrency)
    zarr_store = ObjectStore(obstore_store, read_only=True)
    opened, entries = [], []
    domain = None
    moc_dim = "cells"
    schema_rel = None
    for dec, rel, meta in zip(ancestors, rels, metas):
        stamp = _stamp_from_meta(meta)
        if stamp is None:
            continue  # absent object or unstamped debris (D4)
        if schema_rel is None:
            schema_rel = rel
        node_word = morton_word(dec)
        if index_kind == "moc":
            leaf_domain = MortonRanges.from_shards([node_word], target_order)
            if aoi_words is not None:
                leaf_domain = leaf_domain.intersect(aoi_words)
                if leaf_domain.size == 0:
                    continue
        ds = xr.open_zarr(
            zarr_store,
            group=f"{rel}/{target_order}",
            consolidated=False,
            zarr_format=3,
            **(xr_kwargs or {}),
        )
        _check_composition_fill(ds, rel)
        coords = [name for name in ("morton", "cell_ids") if name in ds]
        ds = ds.set_coords(coords)
        if index_kind == "moc":
            moc_dim = ds["morton"].dims[0] if "morton" in ds.coords else "cells"
            ds = ds.drop_vars(coords)
            if aoi_words is not None:
                full = MortonRanges.from_shards([node_word], target_order)
                ds = ds.isel({moc_dim: full.rank(leaf_domain.fabricate())})
            domain = leaf_domain if domain is None else domain.union(leaf_domain)
        elif aoi_words is not None and "morton" in ds.coords:
            from moczarr.coverage import aoi_mask

            keep = aoi_mask(np.asarray(ds["morton"].values, dtype=np.uint64), aoi_words)
            if not keep.any():
                continue
            ds = ds.isel({ds["morton"].dims[0]: keep})
        attrs = meta.get("attributes") if isinstance(meta, dict) else None
        entries.append(_object_entry(attrs or {}, dec, window if windowed else None, target_order))
        opened.append(ds)
    if schema_rel is None:
        # Declared but no stamped object anywhere at this order/window: not
        # yet swept, or the overviews were deleted (legal — they are D9
        # regenerable caches, never load-bearing). The node is omitted.
        scope = f" window {window!r}" if windowed else ""
        warnings.warn(
            f"declared overview order {k} (cells at order {target_order}) has no "
            f"stamped overview object at {store_root}{scope}; order node omitted "
            f"(not yet swept, or deleted — overviews are regenerable caches)",
            UserWarning,
            stacklevel=2,
        )
        return None
    if not opened:
        # Stamped overviews exist but the AOI excludes them all: the issue-#4
        # posture per node — a schema-correct empty dataset plus a warning.
        warnings.warn(
            f"the given AOI intersects no coverage at {store_root} (overview order "
            f"{target_order}); returning a schema-correct empty dataset (0 cells)",
            UserWarning,
            stacklevel=2,
        )
        ds = xr.open_zarr(
            zarr_store,
            group=f"{schema_rel}/{target_order}",
            consolidated=False,
            zarr_format=3,
            **(xr_kwargs or {}),
        )
        _check_composition_fill(ds, schema_rel)
        coords = [name for name in ("morton", "cell_ids") if name in ds]
        ds = ds.set_coords(coords)
        empty_dim = ds["morton"].dims[0] if "morton" in ds.coords else "cells"
        if index_kind == "moc":
            moc_dim = empty_dim
            ds = ds.drop_vars(coords)
            domain = MortonRanges(np.empty((0, 2), dtype=np.uint64), target_order)
        opened.append(ds.isel({empty_dim: slice(0)}))
    dim = opened[0]["morton"].dims[0] if "morton" in opened[0].coords else "cells"
    if index_kind == "moc":
        dim = moc_dim
    result = xr.concat(opened, dim=dim) if len(opened) > 1 else opened[0]
    if index_kind == "moc":
        from moczarr.moc_index import MortonMocIndex

        assert domain is not None  # an object set it, or the empty path did
        index = MortonMocIndex(domain, dim=dim, name="morton")
        result = result.assign_coords(xr.Coordinates.from_xindex(index))
    if "morton" in result.coords and (
        fabricate_cell_ids is True
        or (fabricate_cell_ids == "auto" and "cell_ids" not in result.coords)
    ):
        from moczarr.fabricate import fabricate_cell_ids as _fabricate

        ids = _fabricate(
            np.asarray(result["morton"].values, dtype=np.uint64),
            level=target_order,
            _stacklevel=4,
        )
        result = result.assign_coords(cell_ids=(result["morton"].dims, ids))
    result = result[sorted(result.data_vars)]
    # The node's own summary: cell_order is THIS order's (decode and
    # fabrication bind level to it); shard_order is the ancestor order the
    # node's objects live at. Source orders ride the per-object entries.
    result.attrs["morton_hive"] = {
        "spec": manifest["spec"],
        "cell_order": target_order,
        "shard_order": k,
        "dataset": manifest.get("dataset"),
    }
    result.attrs[OBJECTS_ATTR] = entries
    if decode:
        from moczarr import dggs

        result = dggs.decode(result, index_kind=index_kind)
    return result


def node_objects(node) -> list[dict]:
    """The per-object entries of one order node (or its dataset).

    Each entry is ``{"node", "window", "role"}`` plus the object's full
    ``zagg_overview`` provenance block when ``role`` is ``"overview"`` —
    the D11 companions (source cell order, per-field aggregation methods)
    live there, per object, never summarized at node level.
    """
    ds = node.ds if hasattr(node, "ds") else node
    return list(ds.attrs.get(OBJECTS_ATTR) or [])


def _orders_with_role(product_node, role: str) -> tuple[int, ...]:
    """Stored cell orders under a product node carrying ``role`` objects."""
    orders = []
    for name, child in product_node.children.items():
        try:
            order = int(name)
        except ValueError:
            continue
        if any(entry.get("role") == role for entry in node_objects(child)):
            orders.append(order)
    return tuple(sorted(orders, reverse=True))


def source_orders(product_node) -> tuple[int, ...]:
    """Every stored cell order carrying at least one *source* object, finest first.

    Defined over a **set** by design (D24): per (shard, window) there is one
    resolution at a time, but heterogeneity is regional across shards, so a
    product may carry source at several orders — "the source node" is not a
    well-formed request. Keys on the per-object ``role`` entries, never on
    node names.
    """
    return _orders_with_role(product_node, "source")


def overview_orders(product_node) -> tuple[int, ...]:
    """Every stored cell order carrying at least one *overview* object, finest first."""
    return _orders_with_role(product_node, "overview")


def finest_source_at(product_node, max_order: int) -> int:
    """The finest source order at or above (coarser than or equal to) ``max_order``.

    "Above" in the pyramid sense: ancestor orders are numerically smaller,
    so this is the largest member of :func:`source_orders` that is
    ``<= max_order``. Raises when no source order qualifies.
    """
    candidates = [order for order in source_orders(product_node) if order <= int(max_order)]
    if not candidates:
        raise ValueError(
            f"no source order at or above (<=) {max_order} under this product "
            f"(source orders present: {list(source_orders(product_node))})"
        )
    return candidates[0]


__all__ = [
    "ALL_TOKEN",
    "OBJECTS_ATTR",
    "OVERVIEW_ATTR",
    "OVERVIEW_SPEC",
    "PYRAMID_SPEC",
    "ROLE_ATTR",
    "finest_source_at",
    "node_objects",
    "open_overview_order",
    "overview_cell_orders",
    "overview_declaration",
    "overview_orders",
    "source_orders",
]

"""Leaf column artifacts (zagg spec §4.6, ``zagg-column/1``) — issue #36.

A **column** is the leaf worker's own pyramid contribution, written at
aggregation time while the shard's cell data is resident: one zarr per
``(leaf, window)``, a sibling of the leaf under the leaf's own node prefix
(``{window stem}.pyramid.zarr`` — the stem from the window ALONE), holding
one **single-chunk, unsharded group per resolution** in the ``{order}/
{field}`` layout — every declared leaf resolution, every within-footprint
rung of the fixed ladder, and the node-order member (the leaf's
whole-footprint aggregate, its universal partial for every coarser cell).
On the published ATL03 store that is groups at orders 13 down to 9.

The reader posture is §4.6's, twice over:

- Columns are derived artifacts a reader **never requires** — absence (or an
  unstamped prefix, debris) is never an error, and the manifest's
  ``pyramid`` block MAY lag what the fleet actually wrote, so *which leaves
  carry columns* is answered by the columns themselves
  (:func:`moczarr.store.walk_columns`, :func:`read_column_record`), not the
  manifest.
- A column MUST NOT be read as a leaf or an overview (the normative
  ``.pyramid.zarr`` name seam); classification is the ``role`` /
  ``zagg_column`` root attrs, checked here.

Reads go through the **normal ragged/dense read path**: :func:`open_column`
returns the same read-only zarr store shape :func:`moczarr.open.open_leaf`
does, so a dense field opens with ``zarr.open_array(store,
path="13/count")`` and a ragged digest streams through
:func:`moczarr.ragged.read_ragged` / :func:`moczarr.ragged.read_cell` at
``"13/h_tdigest_signal"`` — companion channels (``locations=``/``times=``)
included, because a column group carries every sibling its field's §4.5
entry declares, exactly as a leaf does. Full multiscale assembly (a
resolution-aware view over columns + overviews) is deliberately NOT here —
that is espg/moczarr#36b, sequenced behind the #37 data-model ruling.
"""

from __future__ import annotations

from typing import Any

from moczarr.convention import (
    HIVE_SPEC_V2,
    column_path,
    decimal_order,
    manifest_path_grouping,
    morton_decimal,
    parse_manifest,
    validate_window,
)
from moczarr.pyramid import ROLE_ATTR
from moczarr.store import open_object_store, read_manifest

#: Version string of the per-column provenance attrs payload (§4.6).
COLUMN_SPEC = "zagg-column/1"
#: Root-group attrs key carrying the column provenance block, present
#: exactly when ``role`` is ``"column"`` (§4.6).
COLUMN_ATTR = "zagg_column"


def _column_target(
    store_root: str,
    shard: str | int,
    *,
    window: str | None,
    product: str | None,
    manifest: dict | None,
    store: Any,
    store_kwargs: dict[str, Any],
) -> tuple[str, str]:
    """``(subtree root, column rel path)`` with :func:`moczarr.open.open_leaf`'s
    validation: product re-rooting, the window seam, and the shard-order check.
    """
    if product is not None:
        from moczarr.products import validate_product_name

        validate_product_name(product)
        store_root = f"{store_root.rstrip('/')}/{product}"
    if window is not None:
        # The same window seam every non-overview entry point runs through;
        # it is also what refuses the reserved all-time token (#30) — the
        # unwindowed column's `all` stem is spelled window=None.
        validate_window(window, where=store_root)
    if manifest is None:
        manifest = read_manifest(store_root, store=store, **store_kwargs)
        if manifest is None:
            if product is None:
                from moczarr.products import list_products

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
        raise ValueError(
            f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; its columns are "
            f"per-window ({{window}}.pyramid.zarr, D23 naming) — pass window=..."
        )
    order = decimal_order(morton_decimal(shard))
    if order != int(manifest["shard_order"]):
        raise ValueError(
            f"shard {shard!r} is an order-{order} id, but {store_root} shards at order "
            f"{manifest['shard_order']} — columns are named by SHARD id (a cell id "
            f"names no column)"
        )
    rel = column_path(shard, window, path_grouping=manifest_path_grouping(manifest))
    return store_root, rel


def open_column(
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
    """Open ONE shard's §4.6 leaf column as a read-only zarr store.

    The column twin of :func:`moczarr.open.open_leaf` — same parameters,
    same credential/``anonymous`` handling, same validation (the window
    seam, the shard-order check, D19 ``product=`` re-rooting), same
    deliberately bare return: a read-only ``zarr.storage.ObjectStore``
    rooted at the column, for the per-leaf readers at ``"{order}/{field}"``
    paths — ``zarr.open_array`` for dense fields,
    :func:`moczarr.ragged.read_ragged` / :func:`moczarr.ragged.read_cell`
    for ragged digests (companion channels included). The available group
    orders and fields come from :func:`read_column_record`, which also
    answers whether the column exists at all — opening names an object and
    never checks (§4.6 absence is never an error; the first read misses).
    ``store=`` shares a handle for the manifest GET only, exactly as
    ``open_leaf`` does.
    """
    from zarr.storage import ObjectStore

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    store_root, rel = _column_target(
        store_root,
        shard,
        window=window,
        product=product,
        manifest=manifest,
        store=store,
        store_kwargs=store_kwargs,
    )
    column_root = f"{store_root.rstrip('/')}/{rel}"
    return ObjectStore(open_object_store(column_root, **store_kwargs), read_only=True)


def read_column_record(
    store_root: str,
    shard: str | int,
    *,
    window: str | None = None,
    product: str | None = None,
    manifest: dict | None = None,
    anonymous: bool = False,
    store: Any = None,
    **store_kwargs: Any,
) -> dict | None:
    """One shard's validated ``zagg_column`` provenance block, or ``None``.

    One ``zarr.json`` GET (after the manifest's, when not threaded).
    ``None`` means *no committed column*: an absent object, or an unstamped
    prefix (D4 debris — a torn worker, or a declaration that never wrote
    one; §4.6 makes the two indistinguishable from here and neither an
    error, since readers never require a column). A present, **stamped**
    object that does not classify as a column raises: ``role`` must be
    ``"column"``, the :data:`COLUMN_ATTR` block must be present with
    ``spec`` :data:`COLUMN_SPEC` (strict-check, fail loudly — the
    conformance rule; this call names ONE object and cannot half-trust it).

    The record is the §4.6 block verbatim — ``fields`` (the materialized
    fields with their composability classes: the zero-open way to see the
    published ATL03 split between count-only and four-field columns,
    englacial/zagg#547) and ``groups`` (the resolution groups on disk with
    per-group ``regime``/``merges_from_raw``/``n_cells``; orders 9..13 on
    the published ATL03 store) — plus ``node``/``order``/
    ``source_cell_order``/``window``/``cells_with_data_order``. Group
    orders as integers: :func:`column_orders`.
    """
    from moczarr.store import _resolve_store, _stamp_from_meta, read_leaf_metas

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    store_root, rel = _column_target(
        store_root,
        shard,
        window=window,
        product=product,
        manifest=manifest,
        store=store,
        store_kwargs=store_kwargs,
    )
    handle = _resolve_store(store_root, store, store_kwargs)
    meta = read_leaf_metas(store_root, [rel], store=handle)[0]
    if meta is None or _stamp_from_meta(meta) is None:
        return None  # absent, or unstamped debris (D4) — never an error
    attrs = meta.get("attributes") or {}
    role = attrs.get(ROLE_ATTR)
    if role != "column":
        raise ValueError(
            f"stamped object at {rel} declares role {role!r}, not 'column': the §4.6 "
            f"name seam promises a column at this basename, so a stamped non-column "
            f"here is non-conformant (zagg spec §4.6)"
        )
    block = attrs.get(COLUMN_ATTR)
    if not isinstance(block, dict):
        raise ValueError(
            f"column at {rel} lacks the {COLUMN_ATTR!r} provenance block (present "
            f"exactly when role is 'column' — zagg spec §4.6)"
        )
    if block.get("spec") != COLUMN_SPEC:
        raise ValueError(
            f"column at {rel} declares spec {block.get('spec')!r}; this reader "
            f"implements {COLUMN_SPEC!r} only (strict-check, fail loudly)"
        )
    return block


def column_orders(record: dict) -> tuple[int, ...]:
    """A column record's resolution-group cell orders, finest first.

    The integer keys of the §4.6 ``groups`` provenance map — the group set
    actually on disk (every declared leaf resolution, the within-footprint
    ladder rungs, and the node-order member). The finest is the group whose
    populated-cell count the commit stamp records
    (``cells_with_data_order``).
    """
    return tuple(sorted((int(k) for k in record.get("groups") or {}), reverse=True))


__all__ = [
    "COLUMN_ATTR",
    "COLUMN_SPEC",
    "column_orders",
    "open_column",
    "read_column_record",
]

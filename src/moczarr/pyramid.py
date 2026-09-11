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
That mixture is **grammar-supported here but not yet producible**, and the
docstrings below should not be read as coverage of it: this path only ever
names ``{ancestor}/{window}.zarr`` candidates, so a coarse source leaf — named
``{decimal}.zarr`` under the leaf dialect — is never enumerable at an overview
order; and a regionally heterogeneous source domain has no representable root
envelope either, since ``coverage.moc`` carries a single ``order`` and
:func:`moczarr.coverage.ranges_words` raises on any range endpoint off it. So
:func:`source_orders` can gain a second member today only via a role-*stripped*
overview (what the tests construct). The seam that would write a real D24
mixture is spec-legal (§4.3 classifies per object) and unbuilt.

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
    ALL_TOKEN,
    HIVE_SPEC_V2,
    decimal_base,
    manifest_path_grouping,
    morton_decimal,
    morton_word,
    validate_window,
)
from moczarr.coverage import ranges_words
from moczarr.ranges import MortonRanges
from moczarr.store import load_root_coverage, read_leaf_metas

#: Version string of the manifest ``pyramid`` block this reader binds (§4.5).
PYRAMID_SPEC = "zagg-pyramid/1"
#: The §4.5 fixed-ladder revision (englacial/zagg#381/#382): the schedule is
#: the block-level ``overviews`` list — the FULLY EXPANDED ``{node, cells}``
#: level entries, leaf entry first, then every order down to node 0. Bound
#: by the issue #36 declaration surface (:func:`pyramid_declaration`,
#: :func:`read_pyramid`); the order-node *open* path stays ``/1``-only until
#: the #37 data-model ruling (espg/moczarr#36 sequencing).
PYRAMID_SPEC_V2 = "zagg-pyramid/2"
#: Version string of the per-overview provenance attrs payload (§4.3).
OVERVIEW_SPEC = "zagg-overview/1"
#: Root-group attrs key classifying a zarr (D11: per object, never inferred
#: from tree position; source zarrs carry no role — absence means source).
ROLE_ATTR = "role"
#: Root-group attrs key carrying the overview provenance block (§4.3).
OVERVIEW_ATTR = "zagg_overview"
#: Dataset-attrs key carrying one order node's per-object entries.
OBJECTS_ATTR = "zagg_objects"


def overview_declaration(manifest: dict) -> dict | None:
    """The manifest's overview declaration, or ``None`` when declared off.

    This binds the ``/1`` schedule key only: branch on
    ``pyramid.overview.orders`` **first** — an empty ``orders``, a missing
    ``overview`` mapping, or no ``pyramid`` block at all (pre-pyramid
    manifests, and the legacy ``{"orders": []}`` placeholder shape older
    fixtures carry) all mean *no overview family exists* and no other key of
    the block may be assumed. When ``orders`` is non-empty the block's
    ``spec`` is strict-checked (fail loudly on an unknown revision — the
    conformance rule) and ``spacing`` / ``all_time`` / ``fields`` MUST all be
    present; additional keys (``summarize``, the sweep's ``materialized``
    actuals, per-field ``nan_policy``…) are tolerated per the spec.

    A ``zagg-pyramid/2`` block (§4.5, the englacial/zagg#384 default flip:
    the schedule moved to the block-level ``overviews``, and the ``overview``
    family dict carries no legacy ``orders``) therefore reads as ``None``
    here, and the ``/1`` strict-check above is unreachable for it by
    construction. That is a declared-off **view**, never a mis-parse of a
    ``/1`` block: a ``/2`` schedule has no key this function would bind
    wrongly, and §4 makes overviews derived artifacts a reader MUST NOT
    require, so a ``/2`` store's SOURCE data still opens. The ``/2``
    declaration is readable as METADATA through :func:`pyramid_declaration`
    (issue #36); opening a ``/2`` order node stays espg/moczarr#36b/#37, not
    this reader. The degrade is pinned by
    ``tests/test_pyramid.py::TestDeclarationBinding::test_vendored_v2_block_reads_as_no_family``.
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


def pyramid_declaration(manifest: dict) -> dict | None:
    """The manifest's declared pyramid ladder, both grammars, or ``None``.

    The issue #36 declaration surface: where :func:`overview_declaration`
    binds the ``/1`` schedule key for the order-node *open* path (and so
    reads a ``/2`` block as a declared-off view), this decodes the ``pyramid``
    block of **either** revision into one normalized record — metadata only,
    zero I/O. ``None`` means the pyramid is declared off (no block,
    pre-pyramid manifests, the legacy ``{"orders": []}`` placeholder, or the
    canonical §4.5 declared-off form — which is always the ``/1`` shape; a
    ``/2`` ``overviews`` list is never empty). Per §4.5 the reader branches
    on ``spec`` first, then on the revision's schedule key; an unknown
    revision fails loudly (the conformance rule), and the ``/2`` list is
    decoded **verbatim, never re-derived** — the recorded list IS the
    contract (issue #36 / zagg#381).

    The record's keys:

    - ``spec`` — the block's revision string.
    - ``orders`` — the declared **ancestor** orders (``< shard_order``),
      descending: the orders that carry (or will carry) §4.1 overview
      artifacts, uniform across revisions.
    - ``cell_orders`` — ``{ancestor order: [stored cell orders]}``: under
      ``/2`` each entry's ``cells`` verbatim; under ``/1`` the §4.4
      constant-depth cell as a one-member list.
    - ``leaf_cells`` — the ``/2`` leaf entry's ``cells`` (the declared leaf
      resolutions the §4.6 *columns* materialize), ``None`` under ``/1``.
      ``None`` on a ``/2`` block too when the list carries no ``node ==
      shard_order`` entry: §4.6's no-leaf-node-levels arm is a legal
      declaration (the writing run then deletes any column a previous one
      left), so the sentinel reads "this declaration declares no leaf
      resolutions" in **both** grammars rather than distinguishing them —
      and it is deliberately not raised on. Which leaves actually carry
      columns is a question for the columns (:func:`moczarr.store.walk_columns`).
    - ``spacing`` — the ``/1`` schedule step; ``None`` under ``/2`` (the key
      does not exist there).
    - ``overviews`` — the ``/2`` block-level level entries verbatim
      (per-entry ``actuals`` riding along); ``None`` under ``/1``.
    - ``all_time`` / ``exact_levels`` — the family dict's values
      (``exact_levels`` is ``None`` when unwritten — the ``"leaves"`` regime
      writes none).
    - ``fold_source`` — the declared fold regime, defaulted to ``"leaves"``
      when absent (§4.5: the only regime that existed before zagg#376).
    - ``fields`` — the all-fields map verbatim, each entry carrying its
      composability ``class`` (``exact``/``approximate``/``packed``/``none``
      — an unknown token is surfaced as-is; §4.5 tells readers to treat it
      as non-composable rather than error).
    - ``materialized`` — the family-dict sweep actuals verbatim when
      present (on a ``/2`` store that map is the preserved ``/1``-era
      inventory; ``/2`` actuals live per entry in ``overviews``).

    What is *declared* says nothing about what is *on disk* — declaring is
    free, sweeping is the operational decision (zagg#381 point (11)) — so
    the existence question is :func:`read_pyramid`'s probe, and the manifest
    MAY even lag the columns the fleet actually wrote (§4.6: a reader that
    needs to know reads the columns — :func:`moczarr.store.walk_columns`).
    """
    block = manifest.get("pyramid")
    if not isinstance(block, dict):
        return None
    spec = block.get("spec")
    shard_order = int(manifest["shard_order"])
    cell_order = int(manifest["cell_order"])
    overview = block.get("overview")
    if spec == PYRAMID_SPEC_V2:
        entries = block.get("overviews")
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"manifest pyramid block declares {PYRAMID_SPEC_V2!r} but its block-level "
                f"'overviews' schedule is {entries!r}: a /2 list is never empty — the "
                f"declared-off form is always the /1 shape (zagg spec §4.5)"
            )
        if not isinstance(overview, dict):
            raise ValueError(
                f"manifest pyramid block declares {PYRAMID_SPEC_V2!r} without the "
                f"'overview' family dict (zagg spec §4.5)"
            )
        missing = [key for key in ("all_time", "fields") if key not in overview]
        if missing:
            raise ValueError(
                f"manifest pyramid.overview lacks {missing}: with a non-empty schedule, "
                f"all_time/fields MUST be present (zagg spec §4.5)"
            )
        levels: dict[int, list[int]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or "node" not in entry or "cells" not in entry:
                raise ValueError(
                    f"malformed /2 level entry {entry!r}: every member of 'overviews' is "
                    f"a {{node, cells}} mapping with cells a list (zagg spec §4.5)"
                )
            node = int(entry["node"])
            cells = [int(r) for r in entry["cells"]]
            if not (0 <= node <= shard_order) or any(not (node < r < cell_order) for r in cells):
                raise ValueError(
                    f"/2 level entry {entry!r} is off the ladder: node must satisfy "
                    f"0 <= node <= shard_order ({shard_order}) and each cell order "
                    f"node < r < cell_order ({cell_order}) — STRICTLY between, both "
                    f"ends: the r == node group is the §4.6 column's node-order member "
                    f"(a recorded group, never a manifest member) and r == cell_order "
                    f"would BE the base data (zagg spec §4.4/§4.5)"
                )
            if node in levels:
                # Last-writer-wins would make the record self-inconsistent:
                # `overviews` is the VERBATIM surface (it carries the #381
                # point-(7) per-entry `actuals`), so a silently merged
                # duplicate leaves two consumers of the same declaration
                # disagreeing. §4.4 has one level entry — one artifact —
                # per (node, window), so a repeated node is malformed, not
                # a merge instruction.
                raise ValueError(
                    f"/2 'overviews' declares node {node} twice: §4.4 has exactly one "
                    f"level entry (one artifact) per (node, window) (zagg spec §4.4/§4.5)"
                )
            levels[node] = cells
        orders = sorted((k for k in levels if k < shard_order), reverse=True)
        return {
            "spec": spec,
            "orders": orders,
            "cell_orders": {k: levels[k] for k in orders},
            "leaf_cells": levels.get(shard_order),
            "spacing": None,
            "overviews": entries,
            "all_time": overview["all_time"],
            "fold_source": overview.get("fold_source", "leaves"),
            "exact_levels": overview.get("exact_levels"),
            "fields": overview["fields"],
            "materialized": overview.get("materialized"),
        }
    # Unknown revisions fail loudly BEFORE the /1 shape test: a future
    # /3 block might carry no /1 key at all, and reading it as declared-off
    # would be a silent wrong answer (§4.5's conformance rule). ``spec:
    # None`` falls through — the legacy pre-pyramid placeholder carries no
    # spec, and overview_declaration already handles (or refuses) it.
    if spec is not None and spec != PYRAMID_SPEC:
        raise ValueError(
            f"manifest pyramid block declares spec {spec!r}; this reader implements "
            f"{PYRAMID_SPEC!r} and {PYRAMID_SPEC_V2!r} (zagg spec §4.5 — strict-check, "
            f"fail loudly)"
        )
    # /1, and every declared-off shape (which never needs the new grammar,
    # so it is always /1-shaped — including spec-less legacy placeholders).
    decl = overview_declaration(manifest)
    if decl is None:
        return None
    orders = sorted({int(k) for k in decl["orders"]}, reverse=True)
    return {
        "spec": PYRAMID_SPEC,
        "orders": orders,
        "cell_orders": {k: [cell_order - (shard_order - k)] for k in orders},
        "leaf_cells": None,
        "spacing": int(decl["spacing"]),
        "overviews": None,
        "all_time": decl["all_time"],
        "fold_source": decl.get("fold_source", "leaves"),
        "exact_levels": decl.get("exact_levels"),
        "fields": decl["fields"],
        "materialized": decl.get("materialized"),
    }


def read_pyramid(
    store_root: str,
    *,
    product: str | None = None,
    manifest: dict | None = None,
    window: str | None = None,
    orders=None,
    probe: bool = True,
    anonymous: bool = False,
    store: Any = None,
    concurrency: int | None = 32,
    **store_kwargs: Any,
) -> dict | None:
    """A store's declared pyramid ladder plus a cheap materialization report.

    The issue #36 store-root surface: one manifest GET decodes the
    declaration (:func:`pyramid_declaration` — both grammars), and existence
    **probes** answer which declared ancestor orders actually have
    materialized artifacts today. ``None`` when the pyramid is declared off;
    otherwise ``{"declaration": record, "presence": {order: {"nodes": N,
    "stamped": M}} | None}``.

    ``presence`` is per declared ancestor order: ``nodes`` counts the
    candidate ancestor nodes — named *arithmetically* by coarsening the root
    ``coverage.moc``'s source shards (the zagg#201 ruling-(5) enumeration,
    shared with :func:`open_overview_order`) — and ``stamped`` how many hold
    a D4 commit-stamped artifact for this window. One batched ``zarr.json``
    GET per candidate node, never a listing walk and never a data read:
    declared-but-unmaterialized is a **legal recorded state** (declaring is
    free, sweeping is the operational decision — zagg#381 point (11)), so
    ``stamped: 0`` everywhere is an answer, not an error. Presence degrades
    to ``None`` with a warning when the probes cannot be named — no usable
    root ``coverage.moc``, or ``path_grouping > 1`` (the grouped
    ancestor-node path is unsettled writer-side, exactly as
    :func:`open_overview_order` refuses it) — and silently with
    ``probe=False`` (declaration only, one GET total).

    The ``/2`` leaf entry (``declaration["leaf_cells"]``) is materialized by
    the §4.6 **columns**, not by overview artifacts, and the manifest MAY
    lag what the fleet wrote, so presence deliberately covers ancestor
    orders only — read the columns for the leaf tier
    (:func:`moczarr.store.walk_columns`,
    :func:`moczarr.column.read_column_record`).

    ``window`` follows the overview dialect (D23): a windowed
    (``morton-hive/2``) store's artifacts are per window, so probing one
    requires ``window=...``; an unwindowed store refuses the argument (its
    ancestor artifacts are ``all.zarr``). Only that *requirement* is
    probe-scoped — it is about naming objects, and ``probe=False`` names
    none. **Refusals are unconditional**: a label this store has no windows
    for, and the reserved all-time token (``"all"``, §4.2 — three behaviors
    for one token is the espg/moczarr#30 trap :func:`validate_window` exists
    to close), raise at every ``probe``, exactly as
    :func:`open_overview_order` runs the same seam above its windowed
    branch. A ``window=`` this call cannot honour is never silently
    ignored. ``orders`` restricts the
    probe to a subset of the declared ancestor orders (a bounded-cost check
    on a wide ladder). ``product`` re-roots on a D19 multi-product subtree,
    and ``store``/``anonymous``/``store_kwargs`` follow
    :func:`moczarr.open.open_leaf`'s posture (``store`` is rooted at the
    subtree actually read — the product's when ``product`` is given).
    """
    from moczarr.store import _resolve_store, _stamp_from_meta, read_manifest

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    if product is not None:
        from moczarr.products import validate_product_name

        validate_product_name(product)
        store_root = f"{store_root.rstrip('/')}/{product}"
    handle = _resolve_store(store_root, store, store_kwargs)
    if manifest is None:
        manifest = read_manifest(store_root, store=handle)
        if manifest is None:
            raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    else:
        from moczarr.convention import parse_manifest

        manifest = parse_manifest(manifest)
    declaration = pyramid_declaration(manifest)
    if declaration is None:
        return None
    windowed = manifest["spec"] == HIVE_SPEC_V2
    # ABOVE the `probe=False` return, for the reason `open_overview_order`
    # keeps its own `validate_window` above the windowed branch: a `window=`
    # argument this call cannot honour is refused UNCONDITIONALLY, so one
    # store gives one answer about what a window means there. Leaving these
    # under `probe` made the reserved all-time token (#30's trap) and a
    # label on an unwindowed store pass silently whenever probing was off —
    # same argument, same store, two behaviours keyed on an unrelated flag.
    if window is not None:
        validate_window(window, where=store_root)
    if not windowed and window is not None:
        raise ValueError(
            f"window={window!r} on a {manifest['spec']} store: unwindowed stores "
            f"have no window leaves (schedule: none)"
        )
    if not probe:
        return {"declaration": declaration, "presence": None}
    # Genuinely probe-scoped: this one is about NAMING objects, and
    # `probe=False` names none.
    if windowed and window is None:
        raise ValueError(
            f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; its pyramid artifacts "
            f"are per-window (D23 naming) — pass window=... to probe presence, or "
            f"probe=False for the declaration alone"
        )
    probe_orders = declaration["orders"] if orders is None else [int(k) for k in orders]
    bad = [k for k in probe_orders if k not in declaration["orders"]]
    if bad:
        raise ValueError(
            f"orders {bad} are not declared ancestor orders of this pyramid "
            f"(declared: {declaration['orders']})"
        )
    if manifest_path_grouping(manifest) != 1:
        warnings.warn(
            f"{store_root} declares path_grouping > 1: the grouped-tree path of an "
            f"overview ancestor node is not settled writer-side, so presence cannot be "
            f"probed and reads None (the declaration is unaffected; overviews are never "
            f"load-bearing — zagg spec §4.1)",
            UserWarning,
            stacklevel=2,
        )
        return {"declaration": declaration, "presence": None}
    envelope = load_root_coverage(store_root, store=handle)
    if envelope is None:
        warnings.warn(
            f"no usable root coverage.moc at {store_root}: candidate ancestor nodes are "
            f"named by coarsening the source domain, so presence cannot be probed and "
            f"reads None (regenerate the root coverage; the declaration is unaffected)",
            UserWarning,
            stacklevel=2,
        )
        return {"declaration": declaration, "presence": None}
    words = ranges_words(envelope)
    basename = f"{window}.zarr" if windowed else f"{ALL_TOKEN}.zarr"
    per_order = [(k, overview_nodes(manifest, words, k)) for k in probe_orders]
    rels = [f"{_node_rel(dec)}/{basename}" for _k, decs in per_order for dec in decs]
    metas = read_leaf_metas(store_root, rels, store=handle, concurrency=concurrency)
    presence: dict[int, dict[str, int]] = {}
    cursor = 0
    for k, decs in per_order:
        chunk = metas[cursor : cursor + len(decs)]
        cursor += len(decs)
        presence[k] = {
            "nodes": len(decs),
            "stamped": sum(1 for meta in chunk if _stamp_from_meta(meta) is not None),
        }
    return {"declaration": declaration, "presence": presence}


def _ancestor_order(manifest: dict, order: int) -> int:
    """``order`` validated as an ancestor order of the manifest's shard order."""
    shard_order = int(manifest["shard_order"])
    k = int(order)
    if not (0 <= k < shard_order):
        raise ValueError(f"order {k} is not an ancestor order of shard_order {shard_order}")
    return k


def overview_nodes(manifest: dict, shard_words, order: int) -> list[str]:
    """Ancestor-node decimals an overview order names, in packed-word order.

    The zagg#201 ruling-(5) enumeration, as pure arithmetic: the root MOC
    stays the *source* domain, and the objects at declared order ``k`` are
    named by coarsening that domain's shards to their order-``k`` prefix. No
    I/O and no listing — ``shard_words`` is whatever the caller read the root
    ``coverage.moc`` into (:func:`moczarr.coverage.ranges_words`).

    Ordered by PACKED WORD, never by decimal string: a leading ``"-"``
    (southern base cell) sorts before every digit as a string but AFTER every
    northern base cell as a word, so on a store spanning both hemispheres a
    decimal sort lays rows down in an order the §4.4 moc coordinate disagrees
    with. :func:`open_overview_order` and
    :func:`moczarr.stats.read_overview_order_stats` share this function so
    the dataset and telemetry surfaces cannot disagree about which objects an
    order has.
    """
    k = _ancestor_order(manifest, order)
    return sorted(
        {
            dec[: len(decimal_base(dec)) + k]
            for dec in (morton_decimal(int(w)) for w in shard_words)
        },
        key=morton_word,
    )


def _node_rel(decimal: str) -> str:
    """An ancestor node decimal's relative path — one component per digit.

    The writer's convention verbatim (``zagg.sweep._node_rel``, which takes no
    grouping): ``"-311"`` -> ``"-3/1/1"``. Only called on a ``path_grouping:
    1`` store, where it is also the leaf tree's own node path; grouped stores
    are refused in :func:`open_overview_order` rather than guessed at.
    """
    base = decimal_base(decimal)
    return "/".join([base, *decimal[len(base) :]])


def _skip(decimal: str, reason: str) -> None:
    """Warn that one malformed cache object is being dropped (§4.1)."""
    warnings.warn(
        f"overview object at node {decimal} {reason}; the object is dropped and the "
        f"order node keeps its remaining objects (overviews are regenerable caches "
        f"a reader MUST NOT require — zagg spec §4.1; regenerate the sweep)",
        UserWarning,
        stacklevel=4,
    )


def _object_entry(attrs: dict, decimal: str, window: str | None, target_order: int) -> dict | None:
    """One stamped object's per-object entry, or ``None`` to drop the object.

    ``role`` absence means source; ``role: "overview"`` requires an
    interpretable ``zagg_overview`` provenance block (present exactly then,
    a known ``spec``, carrying ``cell_order``) whose ``cell_order`` matches
    this node's order.

    The two severities split on §4.1 — "overviews are **regenerable caches**,
    never load-bearing… a reader MUST NOT require them":

    * **Uninterpretable** — a ``role`` value outside the closed two-value
      vocabulary, a missing or unknown-revision ``zagg_overview`` block, a
      block without ``cell_order``. Warn and skip the object, the posture
      §4.3 already MUSTs for unstamped debris. One malformed cache object
      must not take the node's other objects, let alone the product's
      **source** nodes, down with the whole ``open_store``.
    * **Interpretable but wrong** — a block positively declaring a
      ``cell_order`` other than this node's. Raises: those rows would be
      mis-ranked under this node's §4.4 coordinate, which is a wrong answer
      rather than a missing one, and an off-order fold indicts the sweep
      rather than one object.
    """
    role = attrs.get(ROLE_ATTR)
    entry: dict[str, Any] = {"node": decimal, "window": window, "role": role or "source"}
    if role is None:
        return entry
    if role != "overview":
        _skip(
            decimal,
            f"declares role {role!r}, outside the closed two-value vocabulary "
            f"('overview', or no role key at all — source; zagg spec §4.3)",
        )
        return None
    block = attrs.get(OVERVIEW_ATTR)
    if not isinstance(block, dict):
        _skip(
            decimal,
            f"lacks the {OVERVIEW_ATTR!r} provenance block (present exactly when role "
            f"is 'overview'; zagg spec §4.3)",
        )
        return None
    if block.get("spec") != OVERVIEW_SPEC:
        _skip(
            decimal,
            f"declares spec {block.get('spec')!r}; this reader implements {OVERVIEW_SPEC!r} only",
        )
        return None
    if "cell_order" not in block:
        _skip(decimal, f"has a {OVERVIEW_ATTR!r} block with no 'cell_order' (zagg spec §4.3)")
        return None
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
    _envelope: dict | None = None,
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
    (``"all"``, §4.2) is **refused**. Its ``all.zarr`` folds exist on disk
    (``pyramid.overview.all_time``, §4.5) but have no counterpart on the
    source axis, so surfacing them alone would report a 0-cell source node
    beside overview nodes summing every window — the opt-in surface is
    deferred rather than half-built. On an *unwindowed* store any ``window``
    is refused, the way :func:`moczarr.open.open_hive` refuses it on the
    source axis: the ancestor nodes there hold one ``all.zarr`` apiece, so
    an accepted label would return the all-time rows under a name the store
    has no leaves for. Only ``window=None`` reaches them.

    ``anonymous`` skips request signing for public buckets, and remaining
    ``store_kwargs`` reach ``open_object_store`` — the same posture (and
    spelling) as :func:`moczarr.open.open_hive`, so the remote read path is
    declared in the signature rather than smuggled through ``store_kwargs``.
    ``store`` shares one obstore handle across a product's order nodes and
    the private ``_envelope`` shares that product's already-read root MOC
    (issue #5: one store construction pair and one sidecar read per open, not
    per declared order — how :func:`moczarr.open.open_store` calls this).
    """
    import xarray as xr
    from zarr.storage import ObjectStore

    from moczarr.coverage import as_moc_words
    from moczarr.open import _check_composition_fill
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
    k = _ancestor_order(manifest, order)
    target_order = cell_order - (shard_order - k)
    grouping = manifest_path_grouping(manifest)
    windowed = manifest["spec"] == HIVE_SPEC_V2
    if window is not None:
        # ABOVE the windowed branch, exactly where `open.candidate_leaves`
        # runs it: an unwindowed store's overview basename is `all.zarr`
        # whatever `window=` says, so validating inside `if windowed:` left
        # a `/1` product silently ignoring the argument — the reserved token
        # unrefused, and a real label returning the all-time rows AS IF they
        # were that window (#30's uniformity claim held only for `/2`).
        # `all` passes the label charset, so without this refusal a windowed
        # open would surface the all-time folds while open_hive found no
        # `{shard}_all` leaf and emptied the source node — one tree whose
        # source order reports 0 cells beside overview orders summing EVERY
        # window.
        validate_window(window, where=store_root)
    if windowed:
        if window is None:
            raise ValueError(
                f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; its overview "
                f"orders are per-window (D23 naming) — pass window=..."
            )
        basename = f"{window}.zarr"
    else:
        if window is not None:
            # Same message (and same substance) `open.candidate_leaves`
            # gives for the source axis: one store, one answer about what a
            # window means there.
            raise ValueError(
                f"window={window!r} on a {manifest['spec']} store: unwindowed stores "
                f"have no window leaves (schedule: none)"
            )
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
    envelope = _envelope
    if envelope is None:
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
    aoi_words = as_moc_words(aoi) if aoi is not None else None
    # Candidates are ALL covered ancestors — the AOI scopes rows per object
    # below (and empties to the issue-#4 posture), never the candidate set:
    # tree shape must not depend on the AOI, and the schema for the empty
    # return needs a stamped object to read. Ancestor nodes are 4**(s-k)-fold
    # fewer than shards, so the unscoped stamp GETs stay cheap.
    # Word-ordered, for the reason `overview_nodes` documents: the §4.4 moc
    # coordinate accumulated in `domain` below is word-ascending, and it is
    # the truth for row identity. Same invariant `candidate_leaves` keeps by
    # sorting words before naming leaves (open.py).
    ancestors = overview_nodes(manifest, ranges_words(envelope), k)
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
        # The per-object roster (and its §4.3 validation) is recorded for
        # every STAMPED object, before any AOI filtering: `zagg_objects` is
        # what source_orders/overview_orders answer from, and those are
        # questions about the store's STRUCTURE — the same store must not
        # report "carries no overviews" just because the query's AOI missed
        # them. The AOI governs rows only, exactly as it does for the tree
        # shape (issue #4). This also makes the §4.3 checks AOI-independent.
        attrs = meta.get("attributes") if isinstance(meta, dict) else None
        entry = _object_entry(attrs or {}, dec, window if windowed else None, target_order)
        if entry is None:
            continue  # malformed cache object, warned and dropped (§4.1)
        entries.append(entry)
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
    """Stored cell orders under a product node carrying ``role`` objects.

    A **structural** answer, not a query-scoped one: the per-object entries it
    keys on are recorded for every stamped object the order node names, so an
    ``aoi``/``window`` that empties the rows leaves these answers unchanged.

    Defined on a **flat** (pyramid-declared-off) product node too, where the
    product node itself holds the data instead of an order child: it is source
    at its manifest cell order and carries no overviews. Without that case
    both helpers would answer ``()`` — "no source, no overviews" — for a
    perfectly ordinary product.
    """
    ds = product_node.ds if hasattr(product_node, "ds") else product_node
    native = (ds.attrs.get("morton_hive") or {}).get("cell_order")
    children = {}
    for name, child in product_node.children.items():
        try:
            children[int(name)] = child
        except ValueError:
            continue
    if not children:
        if role != "source" or native is None:
            return ()
        return (int(native),)
    orders = {
        order
        for order, child in children.items()
        if any(entry.get("role") == role for entry in node_objects(child))
    }
    if role == "source" and native is not None and int(native) in children:
        # The leaf level carries source by definition — a manifest fact, not a
        # query result. Its roster CAN legitimately be empty (the root MOC ∩
        # AOI prunes leaf candidates before any stamp is fetched, so a
        # non-intersecting AOI leaves nothing to key on), and dropping the
        # order there would make a structural answer query-dependent.
        orders.add(int(native))
    return tuple(sorted(orders, reverse=True))


def source_orders(product_node) -> tuple[int, ...]:
    """Every stored cell order carrying at least one *source* object, finest first.

    Defined over a **set** by design (D24): per (shard, window) there is one
    resolution at a time, but heterogeneity is regional across shards, so a
    product may carry source at several orders — "the source node" is not a
    well-formed request. Keys on the per-object ``role`` entries, never on
    node names, and answers about the **store**: unaffected by the ``aoi`` or
    ``window`` the tree was opened with. On a flat product node (pyramid
    declared off) the answer is the node's own cell order.
    """
    return _orders_with_role(product_node, "source")


def overview_orders(product_node) -> tuple[int, ...]:
    """Every stored cell order carrying at least one *overview* object, finest first.

    Structural, like :func:`source_orders`: ``()`` means the product stores no
    overviews at all (a flat product node included), never "the query's AOI
    missed them".
    """
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
    "PYRAMID_SPEC_V2",
    "ROLE_ATTR",
    "finest_source_at",
    "node_objects",
    "open_overview_order",
    "overview_cell_orders",
    "overview_declaration",
    "overview_nodes",
    "overview_orders",
    "pyramid_declaration",
    "read_pyramid",
    "source_orders",
]

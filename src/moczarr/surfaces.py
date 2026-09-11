"""Percentile-surface materialization + the resolution ladder (issue #21).

The gridlook feeder: the ratified viewer posture (englacial/zagg#265,
espg/gridlook#1) is that the browser never decodes digest bytes — dense
materialization happens hub-side, and percentile surfaces carry a
``quantile`` DIMENSION so the viewer's existing dimension slider drives
bare-earth↔canopy browsing with zero frontend changes. This module is that
hub-side step, plus the ladder metadata the gridlook#10 order picker
enumerates (declared orders, materialized orders, per-order resolution).

Two surfaces:

- :func:`quantile_surface` / :func:`open_surface` — evaluate a stored
  t-digest field into a dense ``(quantile, cells)`` float array over one
  level's cells, the ``morton`` coordinate carried, the caller's fill
  (default NaN) where a cell holds no digest. Exact/dense fields (a count
  choropleth) ride along un-evaluated via ``carry=``, or alone via
  ``field=None``. Strata-aware: a sequence of digest fields is merged
  per cell before evaluation (one deterministic concatenate-and-sort — the
  lossless t-digest union, no re-compression), so a
  ``h_tdigest_signal``/``h_tdigest_noise`` store answers per-stratum or
  merged-total surfaces by argument.
- :func:`read_ladder` — the picker contract: one typed record per level of
  :func:`moczarr.level.pyramid_levels`' table, extended with the per-order
  ground resolution (``mortie.order2res``) and a probed ``materialized``
  answer (overview presence via :func:`moczarr.pyramid.read_pyramid`,
  column presence via the §4.6 columns' own ``groups`` maps — declared ≠
  materialized is the live-store failure mode the picker must not trip on).

The t-digest algebra is IMPORTED from zagg through the established
``moczarr[zagg]`` extra seam (:func:`moczarr.hhdc._tdigest_algebra`; issue
#19 — vendoring is parity drift by construction), and evaluation honours
the standing trap: centroids are re-sorted by mean whenever an input (a
merged concatenation especially) is not already sorted, because the
interp-based kernel assumes mean order. Native surfaces only (the
englacial/zagg#550 ruling): no ``moczarr.dggs``, no xdggs, no ``decode=``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from moczarr.column import COLUMN_ATTR
from moczarr.convention import (
    HIVE_SPEC_V2,
    column_path,
    manifest_path_grouping,
    morton_decimal,
    validate_window,
)
from moczarr.coverage import ranges_words
from moczarr.level import LEVEL_ATTR, _resolve_target, open_level, pyramid_levels
from moczarr.pyramid import OBJECTS_ATTR, OrderPresence, read_pyramid
from moczarr.store import _stamp_from_meta, load_root_coverage, read_leaf_metas

#: Default quantile set (the STV percentile-slicing conventions, issue #21):
#: 2% ≈ bare earth, 98% ≈ canopy top, 50% ≈ biomass proxy.
DEFAULT_QUANTILES = (0.02, 0.15, 0.50, 0.85, 0.98)

#: Variable-attrs key carrying one surface's evaluation record.
SURFACE_ATTR = "zagg_surface"

#: Dataset attrs a surface carries forward from its input level, verbatim.
_CARRIED_ATTRS = ("morton_hive", LEVEL_ATTR, OBJECTS_ATTR)


def _digest_columns(ds, fields: tuple[str, ...], dim: str) -> list[np.ndarray]:
    """The fields' raw vlen columns, gate-checked as t-digest payloads.

    Each named variable must be an encoded ``zagg-ragged/1`` vlen field
    (dtype ``object`` with the §1.2 attrs block — the shape every level
    Dataset carries digests in) whose element is the t-digest ``(n, 2)``
    (mean, weight) layout. A dense field here is a pointed redirect to the
    ``carry=`` / ``field=None`` arm; a ragged field with another element
    (a ``uint64 (n,)`` locations/times sibling) is refused — evaluating it
    as centroids would be a silent wrong answer.
    """
    from moczarr.ragged import parse_ragged_attrs

    columns = []
    for name in fields:
        if name not in ds:
            raise KeyError(
                f"field {name!r} is not a variable of this level (variables: "
                f"{sorted(ds.data_vars)})"
            )
        var = ds[name]
        if var.dtype != object:
            raise ValueError(
                f"field {name!r} is dense ({var.dtype}), not an encoded ragged digest: "
                f"dense fields need no evaluation — carry them (carry=({name!r},), or "
                f"field=None for a dense-only surface)"
            )
        element = parse_ragged_attrs(var.attrs, field=name)
        if element.inner_shape != (2,):
            raise ValueError(
                f"field {name!r} declares element shape (-1, "
                f"{', '.join(map(str, element.inner_shape))}), not the t-digest "
                f"(-1, 2) (mean, weight) layout — a companion sibling "
                f"(locations/times) is not a digest"
            )
        if var.dims != (dim,):
            raise ValueError(
                f"field {name!r} has dims {var.dims}, expected ({dim!r},): every "
                f"surfaced field must ride the level's cells axis"
            )
        columns.append(np.asarray(var.values, dtype=object))
    return columns


def _cell_digest(columns: list[np.ndarray], elements: list, j: int) -> np.ndarray | None:
    """Cell ``j``'s evaluable digest across ``columns``, or ``None`` when empty.

    Multi-field input is the strata merge: the lossless t-digest union is
    the concatenation of the centroid sets (weights preserved exactly — no
    re-compression). Either way the result is ordered by ``np.lexsort`` on
    ``(weight, mean)`` — unconditionally, so every input walks the same
    code:

    - **by mean**, because the interp-based kernel walks centroids in mean
      order and a concatenation (or any unsorted input) violates that
      silently (the standing unsorted-concatenation trap);
    - **by weight within a mean**, because the kernel's rank walk is
      ``np.cumsum(weights)`` and therefore sensitive to the order of
      centroids that SHARE a mean. A stable sort would keep the argument
      order there, so ``("signal", "noise")`` and the flipped pair would
      disagree whenever the strata happen to share a centroid mean (very
      reachable on quantized elevations). The second key canonicalizes that
      tie independently of the argument order, which is what makes the
      merged total order-independent rather than merely deterministic.
    """
    from moczarr.ragged import decode_cell

    parts = []
    for column, element in zip(columns, elements):
        d = decode_cell(column[j], element)
        if len(d):
            parts.append(d)
    if not parts:
        return None
    digest = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    if len(digest) > 1:
        digest = digest[np.lexsort((digest[:, 1], digest[:, 0]))]
    return digest


def _surface_name(fields: tuple[str, ...], name: str | None) -> str:
    """The output variable name: explicit, the field, or the merged prefix."""
    if name is not None:
        return name
    if len(fields) == 1:
        return fields[0]
    import os.path

    prefix = os.path.commonprefix(list(fields)).rstrip("_")
    return prefix or "merged"


def quantile_surface(
    ds,
    field,
    quantiles=DEFAULT_QUANTILES,
    *,
    fill: float = np.nan,
    carry=(),
    name: str | None = None,
):
    """One level's digest field, dense: a ``(quantile, cells)`` Dataset.

    Takes any Dataset of the issue-#37 level model (:func:`moczarr.level.
    open_level`, :func:`moczarr.open.open_hive`, :func:`moczarr.pyramid.
    open_overview_order`, :func:`moczarr.level.open_column_order` — the
    shape is the model's one law) and evaluates ``field``'s stored
    t-digests per cell at ``quantiles``, returning a NEW dense Dataset:

    - ``field`` as a float64 ``(quantile, cells)`` variable — ``quantile``
      a real dimension with the requested values as its coordinate, so the
      output opens in vanilla xarray and a viewer's dimension slider drives
      it directly;
    - the ``morton`` (and ``cell_ids``, when present) coordinate carried as
      PLAIN arrays — materialization is dense by design, laziness ends
      here;
    - ``fill`` (default NaN) where a cell holds no digest (an absent/empty
      vlen cell, or every stratum empty);
    - ``carry`` names dense (exact-class) variables copied verbatim onto
      the ``cells`` axis — a count choropleth needs no evaluation;
    - the level's ``morton_hive`` / ``zagg_level`` / ``zagg_objects`` attrs
      carried forward, plus :data:`SURFACE_ATTR` on the evaluated variable
      recording ``fields``/``quantiles``/``fill``.

    ``field`` is one digest variable name, a sequence of them (the strata
    arm: the cells' centroid sets are merged by the lossless
    concatenate-and-sort union before evaluation — the merged TOTAL of
    ``("h_tdigest_signal", "h_tdigest_noise")``; per-stratum surfaces are
    one call per field), or ``None`` with a non-empty ``carry`` (the
    dense-only arm: no evaluation, no ``quantile`` dimension). The default
    merged name is the fields' common prefix (``h_tdigest``); ``name=``
    overrides. Evaluation calls zagg's ``quantile_from_tdigest`` per cell —
    the values are exactly that kernel's, per cell, by construction (the
    issue #21 acceptance), imported through the ``moczarr[zagg]`` extra.
    """
    import xarray as xr

    fields: tuple[str, ...]
    if field is None:
        fields = ()
    elif isinstance(field, str):
        fields = (field,)
    else:
        fields = tuple(field)
        if not fields:
            raise ValueError("field=[] names no digest field; pass a name, a sequence, or None")
    if not fields and not carry:
        raise ValueError(
            "field=None materializes no digests, so carry= must name at least one "
            "dense field (the count/exact arm)"
        )

    dim = "cells"
    probe = fields[0] if fields else tuple(carry)[0]
    if probe in ds:
        dims = ds[probe].dims
        if len(dims) != 1:
            raise ValueError(
                f"field {probe!r} has dims {dims}, expected one cells axis: surfaces "
                f"are per-level (one resolution, one 1-D cell domain)"
            )
        dim = dims[0]
    n = int(ds.sizes.get(dim, 0))

    data_vars: dict[str, Any] = {}
    coords: dict[str, Any] = {}
    if fields:
        from moczarr.hhdc import _tdigest_algebra
        from moczarr.ragged import parse_ragged_attrs

        _, quantile_from_tdigest = _tdigest_algebra()
        qs = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
        if qs.ndim != 1 or qs.size == 0:
            raise ValueError(f"quantiles={quantiles!r}: expected a non-empty 1-D sequence")
        if not np.all(np.isfinite(qs)) or qs.min() < 0.0 or qs.max() > 1.0:
            raise ValueError(f"quantiles={quantiles!r}: every quantile must lie in [0, 1]")
        columns = _digest_columns(ds, fields, dim)
        elements = [parse_ragged_attrs(ds[f].attrs, field=f) for f in fields]
        out = np.full((qs.size, n), np.float64(fill), dtype=np.float64)
        for j in range(n):
            digest = _cell_digest(columns, elements, j)
            if digest is None:
                continue
            for i in range(qs.size):
                out[i, j] = quantile_from_tdigest(digest, float(qs[i]))
        out_name = _surface_name(fields, name)
        data_vars[out_name] = xr.DataArray(
            out,
            dims=("quantile", dim),
            attrs={
                SURFACE_ATTR: {
                    "fields": list(fields),
                    "quantiles": qs.tolist(),
                    "fill": float(fill),
                }
            },
        )
        coords["quantile"] = qs

    for cname in carry:
        if cname not in ds:
            raise KeyError(
                f"carry field {cname!r} is not a variable of this level (variables: "
                f"{sorted(ds.data_vars)})"
            )
        var = ds[cname]
        if var.dtype == object:
            raise ValueError(
                f"carry field {cname!r} is an encoded ragged digest: carry= copies "
                f"dense fields verbatim — evaluate a digest by naming it in field="
            )
        if var.dims != (dim,):
            raise ValueError(
                f"carry field {cname!r} has dims {var.dims}, expected ({dim!r},): every "
                f"surfaced field must ride the level's cells axis"
            )
        data_vars[cname] = xr.DataArray(np.asarray(var.values), dims=(dim,), attrs=dict(var.attrs))

    for coord in ("morton", "cell_ids"):
        if coord in ds.coords:
            coords[coord] = (dim, np.asarray(ds[coord].values))
    attrs = {k: ds.attrs[k] for k in _CARRIED_ATTRS if k in ds.attrs}
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def open_surface(
    store_root: str,
    cell_order: int,
    field,
    *,
    quantiles=DEFAULT_QUANTILES,
    fill: float = np.nan,
    carry=(),
    name: str | None = None,
    product: str | None = None,
    manifest: dict | None = None,
    aoi=None,
    window: str | None = None,
    all_time: bool = False,
    anonymous: bool = False,
    concurrency: int | None = 32,
    store: Any = None,
    **store_kwargs: Any,
):
    """Open one level and materialize its surface — the one-call feeder.

    :func:`moczarr.level.open_level` then :func:`quantile_surface`: every
    level-model knob passes through verbatim — ``aoi`` scopes rows exactly
    as the level model does (an out-of-coverage AOI yields the
    schema-correct EMPTY surface, ``(len(quantiles), 0)``), ``window``
    follows the D23 dialect (required on a windowed store, refused on an
    unwindowed one, the reserved all-time token refused everywhere), and
    ``all_time`` reads a windowed store's §4.5 cross-window folds
    (overview levels only). ``None`` — with the arm's own warning — when
    the level is declared but unmaterialized, exactly as ``open_level``
    answers. Surface arguments (``quantiles``/``fill``/``carry``/``name``,
    ``field`` string / sequence / ``None``) follow
    :func:`quantile_surface`.
    """
    ds = open_level(
        store_root,
        cell_order,
        product=product,
        manifest=manifest,
        aoi=aoi,
        window=window,
        all_time=all_time,
        anonymous=anonymous,
        concurrency=concurrency,
        store=store,
        **store_kwargs,
    )
    if ds is None:
        return None
    return quantile_surface(ds, field, quantiles, fill=fill, carry=carry, name=name)


@dataclass(frozen=True)
class LadderLevel:
    """One resolution level of the picker's ladder (:func:`read_ladder`).

    Attributes
    ----------
    cell_order : int
        The level's resolution — the cell order a reader picks
        (:func:`moczarr.level.pyramid_levels`' key, and
        :func:`moczarr.level.open_level` / :func:`open_surface`'s address).
    order : int
        The hive-tree node order whose artifact carries the level.
    artifact : str
        ``"source"`` | ``"column"`` | ``"overview"`` — the level table's
        vocabulary.
    resolution_km : float
        Approximate ground scale of one cell at ``cell_order``
        (``mortie.order2res``: the RMS cell spacing on the equal-area
        HEALPix sphere), so catalog labels and camera→order mappings are
        derived, never hardcoded.
    materialized : bool or None
        Whether the level has stamped artifacts on disk today: ``True`` for
        the source level (the leaves ARE the store), probed for overview
        and column levels, ``None`` when unprobed (``probe=False``) or
        unprobeable (no usable root coverage; a grouped tree's overview
        nodes). Declared ≠ materialized is a legal recorded state — the
        picker lists ``True`` levels only.
    presence : OrderPresence or None
        The probe counts behind ``materialized``, when probed: for an
        overview level the D4 existence probe (:func:`moczarr.pyramid.
        read_pyramid`'s numbers for its node order); for a column level
        ``nodes`` counts the candidate leaves and ``stamped`` the admitted
        columns whose own ``groups`` map carries this resolution
        (classification-bound, unlike the overview probe — a column's
        membership lives only in its attrs). ``None`` on the source level
        and whenever unprobed.
    """

    cell_order: int
    order: int
    artifact: str
    resolution_km: float
    materialized: bool | None
    presence: OrderPresence | None = None


@dataclass(frozen=True)
class Ladder:
    """A store's resolution ladder, typed — the gridlook#10 picker contract.

    ``levels`` is the whole declared table, finest first; the two
    projections answer the picker's questions directly: ``declared`` — every
    addressable cell order — and ``materialized`` — the subset with stamped
    artifacts today (an order declared but not materialized never becomes a
    catalog entry). Per-order resolution rides each :class:`LadderLevel`.
    """

    levels: tuple[LadderLevel, ...]

    @property
    def declared(self) -> tuple[int, ...]:
        """Every declared cell order, finest first."""
        return tuple(lvl.cell_order for lvl in self.levels)

    @property
    def materialized(self) -> tuple[int, ...]:
        """The declared cell orders with stamped artifacts today (probed)."""
        return tuple(lvl.cell_order for lvl in self.levels if lvl.materialized)


def _column_presence(
    store_root: str,
    manifest: dict,
    leaf_cells: list[int],
    window: str | None,
    handle: Any,
    concurrency: int | None,
) -> dict[int, OrderPresence] | None:
    """Per column level: how many candidate leaves' columns carry the group.

    The leaf-tier half of the ladder probe, which :func:`moczarr.pyramid.
    read_pyramid` deliberately leaves out (§4.6: the manifest MAY lag what
    the fleet wrote, so column membership is answerable only from the
    columns' own ``groups`` maps). One batched ``zarr.json`` GET per
    candidate leaf — the same enumeration and classification
    :func:`moczarr.level.open_column_order` runs (:func:`moczarr.level.
    _column_entry`'s two severities), shared here across every column level
    so the ladder pays the leaf tier once, not once per level. ``None``
    (with a warning) when the root coverage is unusable — candidates are
    named arithmetically, exactly as the openers refuse to walk.
    """
    from moczarr.level import _column_entry

    envelope = load_root_coverage(store_root, store=handle)
    if envelope is None:
        warnings.warn(
            f"no usable root coverage.moc at {store_root}: candidate columns are named "
            f"arithmetically from the source domain, so column materialization cannot "
            f"be probed and reads None (regenerate the root coverage)",
            UserWarning,
            stacklevel=3,
        )
        return None
    grouping = manifest_path_grouping(manifest)
    shard_order = int(manifest["shard_order"])
    words = np.sort(ranges_words(envelope))
    shards = [morton_decimal(int(w)) for w in words]
    rels = [column_path(dec, window, path_grouping=grouping) for dec in shards]
    metas = read_leaf_metas(store_root, rels, store=handle, concurrency=concurrency)
    carrying: dict[int, int] = {r: 0 for r in leaf_cells}
    for dec, meta in zip(shards, metas):
        if _stamp_from_meta(meta) is None:
            continue  # no column, or unstamped debris (D4/§4.6) — never an error
        attrs = meta.get("attributes") if isinstance(meta, dict) else None
        entry = _column_entry(attrs or {}, dec, window, shard_order)
        if entry is None:
            continue  # malformed artifact, warned and dropped
        groups = entry[COLUMN_ATTR].get("groups") or {}
        for r in leaf_cells:
            if str(r) in groups:
                carrying[r] += 1
    return {r: OrderPresence(nodes=len(shards), stamped=carrying[r]) for r in leaf_cells}


def read_ladder(
    store_root: str,
    *,
    product: str | None = None,
    manifest: dict | None = None,
    window: str | None = None,
    probe: bool = True,
    anonymous: bool = False,
    concurrency: int | None = 32,
    store: Any = None,
    **store_kwargs: Any,
) -> Ladder:
    """A store's resolution ladder as one typed record — the picker feeder.

    The gridlook#10 Phase-3 contract in one call: **declared orders** (the
    level table, both grammars — a declared-off pyramid is the one-level
    ladder), **materialized orders** (probed: which declared levels have
    stamped artifacts on disk today — declared ≠ materialized is the
    live-store failure mode a picker must not trip on), and **per-order
    resolution** (``mortie.order2res`` at each cell order). Mostly a
    projection of :func:`moczarr.level.pyramid_levels` +
    :func:`moczarr.pyramid.read_pyramid`, typed per the
    :class:`~moczarr.pyramid.PyramidInfo` posture.

    ``materialized`` per artifact kind: the source level is ``True`` (the
    leaves ARE the store); overview levels take :func:`moczarr.pyramid.
    read_pyramid`'s D4 existence probe (``stamped > 0`` — existence, not
    readability, its documented split); column levels are probed from the
    §4.6 columns' own ``groups`` maps (one batched GET per candidate leaf,
    shared across the column levels — see :func:`_column_presence`), since
    the manifest MAY lag the fleet there. ``None`` marks *unknown*:
    ``probe=False`` (declaration only — one manifest GET, no coverage or
    stamp reads), an unusable root coverage, or the grouped-tree overview
    refusal — :attr:`Ladder.materialized` lists ``True`` levels only, so an
    unknown level never masquerades as materialized.

    ``window`` follows the shared seam: probing a windowed
    (``morton-hive/2``) store requires ``window=...`` (its artifacts are
    per window — D23), an unwindowed store refuses the argument, and the
    reserved all-time token is refused everywhere. The windowed store's
    §4.5 all-time fold tier is NOT probed here (its levels are the same
    cell orders; open them with ``all_time=True``). ``product`` re-roots on
    a D19 multi-product subtree; ``manifest``/``store`` thread an
    already-read manifest and handle, per the issue-#5 posture.
    """
    from mortie import order2res

    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    store_root, handle, manifest = _resolve_target(
        store_root, product, manifest, store, store_kwargs
    )
    table = pyramid_levels(manifest)
    windowed = manifest["spec"] == HIVE_SPEC_V2
    if window is not None:
        validate_window(window, where=store_root)
    if not windowed and window is not None:
        raise ValueError(
            f"window={window!r} on a {manifest['spec']} store: unwindowed stores "
            f"have no window leaves (schedule: none)"
        )
    needs_probe = probe and any(rec["artifact"] != "source" for rec in table.values())
    if needs_probe and windowed and window is None:
        raise ValueError(
            f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; its pyramid artifacts "
            f"are per-window (D23 naming) — pass window=... to probe materialization, "
            f"or probe=False for the declared ladder alone"
        )

    overview_presence: dict[int, OrderPresence] | None = None
    column_presence: dict[int, OrderPresence] | None = None
    if needs_probe:
        if any(rec["artifact"] == "overview" for rec in table.values()):
            info = read_pyramid(
                store_root,
                manifest=manifest,
                window=window,
                store=handle,
                concurrency=concurrency,
            )
            overview_presence = info.presence if info is not None else None
        leaf_cells = sorted(
            (rec["cell_order"] for rec in table.values() if rec["artifact"] == "column"),
            reverse=True,
        )
        if leaf_cells:
            column_presence = _column_presence(
                store_root, manifest, leaf_cells, window, handle, concurrency
            )

    levels = []
    for rec in table.values():
        presence: OrderPresence | None = None
        if rec["artifact"] == "source":
            materialized: bool | None = True
        elif not needs_probe:
            materialized = None
        elif rec["artifact"] == "overview":
            presence = (overview_presence or {}).get(rec["order"])
            materialized = presence.stamped > 0 if presence is not None else None
        else:
            presence = (column_presence or {}).get(rec["cell_order"])
            materialized = presence.stamped > 0 if presence is not None else None
        levels.append(
            LadderLevel(
                cell_order=rec["cell_order"],
                order=rec["order"],
                artifact=rec["artifact"],
                resolution_km=float(order2res(rec["cell_order"])),
                materialized=materialized,
                presence=presence,
            )
        )
    return Ladder(levels=tuple(levels))


__all__ = [
    "DEFAULT_QUANTILES",
    "SURFACE_ATTR",
    "Ladder",
    "LadderLevel",
    "open_surface",
    "quantile_surface",
    "read_ladder",
]

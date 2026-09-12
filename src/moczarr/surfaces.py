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
  merged-total surfaces by argument — and by argument ONLY: the sort key
  is ``(mean, weight)``, so naming the strata the other way round is the
  same surface even when they share a centroid mean.
- :func:`read_ladder` — the picker contract: one typed record per level of
  :func:`moczarr.level.pyramid_levels`' table, extended with the per-order
  ground resolution (``mortie.order2res``) and a probed ``materialized``
  answer (overview presence via :func:`moczarr.pyramid.read_pyramid`,
  column presence via the §4.6 columns' own ``groups`` maps — declared ≠
  materialized is the live-store failure mode the picker must not trip on).

The t-digest algebra is IMPORTED from zagg through the established
``moczarr[zagg]`` extra seam (:func:`moczarr.hhdc._tdigest_algebra`; issue
#19 — vendoring is parity drift by construction), and evaluation honours
the standing trap: every digest — merged concatenation or single stored
cell — is ordered by ``(mean, weight)`` before the kernel sees it, because
the interp-based kernel assumes mean order AND its cumulative-weight walk
is sensitive to the order of centroids that share a mean. Native surfaces
only (the
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
    leaf_path,
    manifest_path_grouping,
    morton_decimal,
    validate_window,
)
from moczarr.coverage import ranges_words
from moczarr.level import LEVEL_ATTR, _resolve_target, open_level, pyramid_levels
from moczarr.pyramid import OBJECTS_ATTR, OrderPresence, pyramid_declaration, read_pyramid
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


def _json_fill(fill: float) -> float | str:
    """``fill`` as a STRICT-JSON attrs value: a number, or the token's name.

    A non-finite fill (the ``np.nan`` default, above all) recorded as a bare
    float writes the literal ``NaN`` into the variable's ``zarr.json`` —
    which ``json`` accepts (``allow_nan=True``) and ``JSON.parse`` does not,
    taking the WHOLE metadata object down with it (shape, dtype and chunk
    grid included) in the browser this module exists to feed. The house
    spelling is the quoted name — zarr's own ``"fill_value": "NaN"`` and the
    manifest's fill entries — so non-finite fills are recorded that way.
    """
    value = float(fill)
    if np.isfinite(value):
        return value
    if np.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


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
      the ``cells`` axis — a count choropleth needs no evaluation. One
      name, a sequence of them, or any iterable (consumed once, like
      ``field``: a bare string is ONE name, never its characters);
    - the level's ``morton_hive`` / ``zagg_level`` / ``zagg_objects`` attrs
      carried forward, plus :data:`SURFACE_ATTR` on the evaluated variable
      recording ``fields``/``quantiles``/``fill`` — the last as a number,
      or as the quoted token name (``"NaN"``/``"Infinity"``/
      ``"-Infinity"``) when it is not finite, so the written ``zarr.json``
      stays strict JSON for a browser reader (:func:`_json_fill`).

    ``field`` is one digest variable name, a sequence of them (the strata
    arm: the cells' centroid sets are merged by the lossless
    concatenate-and-sort union before evaluation — the merged TOTAL of
    ``("h_tdigest_signal", "h_tdigest_noise")``, identical to the flipped
    pair's, ties included; per-stratum surfaces are
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
    # Normalized ONCE, beside field's: a bare string is one name (not its
    # characters), and a one-shot iterable must survive both the probe below
    # and the carry loop at the bottom.
    carry = (carry,) if isinstance(carry, str) else tuple(carry)
    if not fields and not carry:
        raise ValueError(
            "field=None materializes no digests, so carry= must name at least one "
            "dense field (the count/exact arm)"
        )

    dim = "cells"
    probe = fields[0] if fields else carry[0]
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
                    "fill": _json_fill(fill),
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
    fabricate_cell_ids: bool | str = "auto",
    index_kind: str = "moc",
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    store: Any = None,
    **store_kwargs: Any,
):
    """Open one level and materialize its surface — the one-call feeder.

    :func:`moczarr.level.open_level` then :func:`quantile_surface`, with
    ``open_level``'s whole signature threaded through verbatim (defaults
    unchanged): ``aoi`` scopes rows exactly as the level model does (an
    out-of-coverage AOI yields the schema-correct EMPTY surface,
    ``(len(quantiles), 0)``), ``window`` follows the D23 dialect (required
    on a windowed store, refused on an unwindowed one, the reserved
    all-time token refused everywhere), ``all_time`` reads a windowed
    store's §4.5 cross-window folds (overview levels only),
    ``fabricate_cell_ids`` governs whether the surface carries a
    ``cell_ids`` coordinate at all, ``index_kind`` selects the level's
    index (and with it how ``aoi`` cuts rows), and ``xr_kwargs`` reaches
    ``xr.open_zarr`` — the knob that governs the read, which is the wall at
    viewer scales. ``None`` — with the arm's own warning — when the level
    is declared but unmaterialized, exactly as ``open_level`` answers.
    Surface arguments (``quantiles``/``fill``/``carry``/``name``, ``field``
    string / sequence / ``None``) follow :func:`quantile_surface`.
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
        fabricate_cell_ids=fabricate_cell_ids,
        index_kind=index_kind,
        concurrency=concurrency,
        xr_kwargs=xr_kwargs,
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
        the source level of an unwindowed store (the leaves ARE the store),
        probed for overview and column levels — and for the source level
        too once a windowed store is read at a NAMED window, whose leaves
        are per window (``{shard}_{window}.zarr``, D23) and exist only for
        the windows that were swept. ``None`` when unprobed
        (``probe=False``) or unprobeable (no usable root coverage; a
        grouped tree's overview nodes; an overview rung whose ``/2`` node
        declares SEVERAL cell orders, which the node-existence probe cannot
        tell apart). Declared ≠ materialized is a legal recorded state —
        the picker lists ``True`` levels only.
    presence : OrderPresence or None
        The probe counts behind ``materialized``, when probed: for an
        overview level the D4 existence probe (:func:`moczarr.pyramid.
        read_pyramid`'s numbers for its node order); for a column level
        ``nodes`` counts the candidate leaves and ``stamped`` the admitted
        columns whose own ``groups`` map carries this resolution
        (classification-bound, unlike the overview probe — a column's
        membership lives only in its attrs); for a windowed store's probed
        source level ``nodes`` counts the candidate leaves and ``stamped``
        the ones stamped for that window. ``None`` on an unprobed level and
        on an unwindowed store's source level, which is not probed.
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


def _leaf_presence(
    store_root: str,
    manifest: dict,
    leaf_cells: list[int],
    window: str | None,
    handle: Any,
    concurrency: int | None,
    *,
    source: bool,
) -> tuple[OrderPresence | None, dict[int, OrderPresence] | None]:
    """The leaf tier's presence: the source leaves, the column levels, or both.

    The leaf-tier half of the ladder probe, which :func:`moczarr.pyramid.
    read_pyramid` deliberately leaves out (§4.6: the manifest MAY lag what
    the fleet wrote, so column membership is answerable only from the
    columns' own ``groups`` maps). Both halves ride ONE batched
    ``zarr.json`` read over the same arithmetically named candidates, so
    the ladder pays the leaf tier once — not once per level, and not twice
    when a windowed store wants its source rung probed too:

    - ``source`` counts the leaves themselves (``{shard}_{window}.zarr`` —
      D23), the existence question :func:`read_ladder` asks only when a
      windowed store's per-window leaves make ``True`` a guess. The D4
      stamp is the same existence test the overview probe uses.
    - ``leaf_cells`` counts, per declared leaf resolution, the admitted
      columns whose own ``groups`` map carries it — the same enumeration
      and classification :func:`moczarr.level.open_column_order` runs
      (:func:`moczarr.level._column_entry`).

    ``(None, None)`` (with a warning) when the root coverage is unusable —
    candidates are named arithmetically, exactly as the openers refuse to
    walk.

    NEITHER of ``_column_entry``'s severities is fatal here: this is a
    presence count, not a read. An uninterpretable object is warned and
    dropped by ``_column_entry`` itself; a mis-stamped one (its wrong-
    identity ``ValueError``) is caught, warned about by name, and counted
    as not carrying any declared resolution — a column declaring another
    node is evidence of absence at this leaf. The severity stays exactly as
    it is for :func:`moczarr.level.open_column_order`, which would be about
    to read those groups under the wrong node's identity.
    """
    from moczarr.level import _column_entry

    envelope = load_root_coverage(store_root, store=handle)
    if envelope is None:
        warnings.warn(
            f"no usable root coverage.moc at {store_root}: candidate leaves and columns "
            f"are named arithmetically from the source domain, so leaf-tier "
            f"materialization cannot be probed and reads None (regenerate the root "
            f"coverage)",
            UserWarning,
            stacklevel=3,
        )
        return None, None
    grouping = manifest_path_grouping(manifest)
    shard_order = int(manifest["shard_order"])
    words = np.sort(ranges_words(envelope))
    shards = [morton_decimal(int(w)) for w in words]
    rels = []
    if source:
        rels += [leaf_path(dec, window, path_grouping=grouping) for dec in shards]
    if leaf_cells:
        rels += [column_path(dec, window, path_grouping=grouping) for dec in shards]
    metas = read_leaf_metas(store_root, rels, store=handle, concurrency=concurrency)
    source_presence: OrderPresence | None = None
    if source:
        stamped = sum(_stamp_from_meta(m) is not None for m in metas[: len(shards)])
        source_presence = OrderPresence(nodes=len(shards), stamped=stamped)
        metas = metas[len(shards) :]
    if not leaf_cells:
        return source_presence, None
    carrying: dict[int, int] = {r: 0 for r in leaf_cells}
    for dec, meta in zip(shards, metas):
        if _stamp_from_meta(meta) is None:
            continue  # no column, or unstamped debris (D4/§4.6) — never an error
        attrs = meta.get("attributes") if isinstance(meta, dict) else None
        try:
            entry = _column_entry(attrs or {}, dec, window, shard_order)
        except ValueError as exc:
            # _column_entry's OTHER severity: a block positively declaring
            # another leaf's node/order/window. That severity is right for
            # open_column_order, which is about to read those groups under
            # the wrong node's identity — a wrong answer. It is not right
            # for a presence COUNT: a column declaring another node is
            # evidence of this leaf's ABSENCE, so it counts as not-carrying
            # and the ladder's other rungs (source, overview) stay
            # answerable. One mis-stamped object never takes the picker's
            # whole catalog entry down.
            warnings.warn(
                f"column at node {dec} is mis-stamped and counts as not carrying any "
                f"declared leaf resolution ({exc}); open_column_order still refuses it",
                UserWarning,
                stacklevel=3,
            )
            continue
        if entry is None:
            continue  # malformed artifact, warned and dropped
        groups = entry[COLUMN_ATTR].get("groups") or {}
        for r in leaf_cells:
            if str(r) in groups:
                carrying[r] += 1
    return source_presence, {
        r: OrderPresence(nodes=len(shards), stamped=carrying[r]) for r in leaf_cells
    }


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

    ``materialized`` per artifact kind: overview levels take
    :func:`moczarr.pyramid.read_pyramid`'s D4 existence probe (``stamped >
    0`` — existence, not readability, its documented split) — except where
    a ``/2`` node declares SEVERAL cell orders, whose rungs read ``None``:
    the node's object carries a scalar ``zagg_overview.cell_order``, so at
    most one of those rungs is backed and a node-existence count cannot say
    which; column levels
    are probed from the §4.6 columns' own ``groups`` maps (one batched GET
    per candidate leaf, shared across the column levels — see
    :func:`_leaf_presence`), since the manifest MAY lag the fleet there.
    The **source** level is ``True`` by definition on an unwindowed store
    (the leaves ARE the store) and is PROBED on a windowed one whenever a
    window is named: those leaves are per window (``{shard}_{window}.zarr``
    — D23) and exist only for the windows that were swept, so a
    syntactically valid label that was never swept must read ``False``, not
    ``True``. It rides the same batched leaf-tier read as the columns, so
    the extra honesty costs no extra round trip when a column level is
    probed too — and on a windowed store with no column levels it is the
    one leaf-tier batch the ladder pays (``probe=False`` opts out
    entirely). The probe never raises: a malformed or mis-stamped column is
    warned about and counted as not carrying, so one bad object cannot take
    the picker's whole catalog entry — the source and overview rungs
    included — down with it. ``None`` marks *unknown*: ``probe=False``
    (declaration only — one manifest GET, no coverage or stamp reads, so a
    windowed store's source rung reads unknown rather than guessing at a
    named window), an unusable root coverage, or the grouped-tree overview
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

    # The source rung is True by definition on an unwindowed store — the
    # leaves ARE the store — but a windowed store's leaves are per window
    # (D23), so once a window is NAMED the same declared-vs-materialized
    # question applies to it as to every other rung, and it is probed.
    source_probe = probe and windowed and window is not None

    overview_presence: dict[int, OrderPresence] | None = None
    column_presence: dict[int, OrderPresence] | None = None
    source_presence: OrderPresence | None = None
    if needs_probe or source_probe:
        if needs_probe and any(rec["artifact"] == "overview" for rec in table.values()):
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
        if not needs_probe:
            leaf_cells = []
        if leaf_cells or source_probe:
            source_presence, column_presence = _leaf_presence(
                store_root,
                manifest,
                leaf_cells,
                window,
                handle,
                concurrency,
                source=source_probe,
            )

    decl = pyramid_declaration(manifest)
    multi_cell_nodes = {
        int(k) for k, cells in ((decl or {}).get("cell_orders") or {}).items() if len(cells) > 1
    }

    levels = []
    for rec in table.values():
        presence: OrderPresence | None = None
        if rec["artifact"] == "source":
            materialized: bool | None
            if not (windowed and window is not None):
                materialized = True  # the leaves ARE the store
            elif not probe:
                materialized = None  # a named window, declaration only
            else:
                presence = source_presence
                materialized = presence.stamped > 0 if presence is not None else None
        elif not needs_probe:
            materialized = None
        elif rec["artifact"] == "overview" and rec["order"] in multi_cell_nodes:
            # A /2 node declaring SEVERAL cell orders is backed by at most
            # one of them — its object's zagg_overview.cell_order is scalar
            # — and read_pyramid's probe is a bare node-existence count that
            # says nothing about which. Fanning that one count out to every
            # declared rung would tell the picker two orders are
            # materialized and then raise on the second (open_level names
            # the level explicitly, and _object_entry's wrong-cell_order
            # severity is a raise). Unknown is the honest answer the
            # existence probe can give.
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

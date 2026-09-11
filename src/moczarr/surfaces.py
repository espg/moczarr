"""Percentile-surface materialization: the gridlook feeder (issue #21).

The ratified viewer posture (englacial/zagg#265, espg/gridlook#1) is that
the browser never decodes digest bytes — dense materialization happens
hub-side, and percentile surfaces carry a ``quantile`` DIMENSION so the
viewer's existing dimension slider drives bare-earth↔canopy browsing with
zero frontend changes. This module is that hub-side step:
:func:`quantile_surface` / :func:`open_surface` evaluate a stored t-digest
field into a dense ``(quantile, cells)`` float array over one level's
cells, the ``morton`` coordinate carried, the caller's fill (default NaN)
where a cell holds no digest. Exact/dense fields (a count choropleth) ride
along un-evaluated via ``carry=``, or alone via ``field=None``.
Strata-aware: a sequence of digest fields is merged per cell before
evaluation (one deterministic concatenate-and-sort — the lossless t-digest
union, no re-compression), so a ``h_tdigest_signal``/``h_tdigest_noise``
store answers per-stratum or merged-total surfaces by argument.

The t-digest algebra is IMPORTED from zagg through the established
``moczarr[zagg]`` extra seam (:func:`moczarr.hhdc._tdigest_algebra`; issue
#19 — vendoring is parity drift by construction), and evaluation honours
the standing trap: centroids are re-sorted by mean whenever an input (a
merged concatenation especially) is not already sorted, because the
interp-based kernel assumes mean order. Native surfaces only (the
englacial/zagg#550 ruling): no ``moczarr.dggs``, no xdggs, no ``decode=``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from moczarr.level import LEVEL_ATTR, open_level
from moczarr.pyramid import OBJECTS_ATTR

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
    re-compression, so the merged total is deterministic and
    order-independent). Either way the result is re-sorted by mean when it
    is not already sorted — the interp-based kernel walks centroids in mean
    order, and a concatenation (or any unsorted input) violates that
    silently (the standing unsorted-concatenation trap).
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
    means = digest[:, 0]
    if means.size > 1 and np.any(np.diff(means) < 0):
        digest = digest[np.argsort(means, kind="stable")]
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


__all__ = [
    "DEFAULT_QUANTILES",
    "SURFACE_ATTR",
    "open_surface",
    "quantile_surface",
]

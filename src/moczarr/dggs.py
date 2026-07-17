"""xdggs integration: ``grid_name: "morton"`` in the public grid registry.

The layer-2 extra (``pip install 'moczarr[xdggs]'``): a :class:`MortonInfo` /
:class:`MortonIndex` pair registered via ``register_dggs("morton")``, so the
``ds.dggs`` accessor (``sel_latlon``, ``cell_centers``, ``cell_boundaries``,
``zoom_to``) works on the ``morton`` coordinate of an ``open_hive`` result.
Importing this module performs the registration; the core package never
imports it (xdggs and its heavy deps stay optional).

Cell ids here are mortie's packed ``uint64`` morton words — NOT bare HEALPix
nested ids (those ride along as the unindexed ``cell_ids`` coordinate). A
word encodes its own order, so transforms that read cells (``mort2geo``,
``mort2polygon``, ``clip2order``) never consult ``level``; only the binning
direction (``geographic2cell_ids``) and child generation do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import xarray as xr

try:
    from xdggs.grid import DGGSInfo, translate_parameters
    from xdggs.healpix import center_around_prime_meridian
    from xdggs.index import DGGSIndex
    from xdggs.utils import _extract_cell_id_variable, register_dggs
except ImportError as exc:  # pragma: no cover - exercised only in core-only envs
    raise ImportError(
        "moczarr.dggs requires xdggs; install the extra: pip install 'moczarr[xdggs]'"
    ) from exc

GRID_NAME = "morton"


def _words(cell_ids) -> np.ndarray:
    """Cell ids as a contiguous ``uint64`` array (mortie's kernel input)."""
    return np.ascontiguousarray(cell_ids, dtype=np.uint64)


@dataclass(frozen=True)
class MortonInfo(DGGSInfo):
    """Grid info for packed-morton HEALPix cells (mortie encoding).

    ``level`` is the HEALPix order of the cells — ``cell_order`` in the
    morton-hive manifest (:func:`decode` translates one to the other). The
    packing supports orders 0..29 (2 bits per level in a ``uint64`` word).
    """

    level: int

    valid_parameters: ClassVar[dict[str, Any]] = {"level": range(30)}

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer between 0 and 29")

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> "MortonInfo":
        """Construct from grid attrs; accepts ``level`` or manifest ``cell_order``."""
        translations = {
            "level": ("level", int),
            "cell_order": ("level", int),
        }
        params = translate_parameters(mapping, translations)
        return cls(**params)

    def to_dict(self) -> dict[str, Any]:
        """The normalized grid parameters (the xdggs coordinate-attrs form)."""
        return {"grid_name": GRID_NAME, "level": self.level}

    def cell_ids2geographic(self, cell_ids) -> tuple[np.ndarray, np.ndarray]:
        """``(lon, lat)`` cell centers in degrees, lon normalized to [-180, 180).

        ``mort2geo`` reads each word's own order — ``level`` is not consulted,
        but mortie rejects mixed-order input per call — and returns lon in
        [0, 360); the xdggs surface is [-180, 180).
        """
        from mortie import mort2geo

        lat, lon = mort2geo(_words(cell_ids))
        lon = (np.atleast_1d(np.asarray(lon, dtype=np.float64)) + 180.0) % 360.0 - 180.0
        return lon, np.atleast_1d(np.asarray(lat, dtype=np.float64))

    def geographic2cell_ids(self, lon, lat) -> np.ndarray:
        """Bin points (degrees) to the morton words containing them at ``level``."""
        from mortie import geo2mort

        return geo2mort(
            np.atleast_1d(np.asarray(lat, dtype=np.float64)),
            np.atleast_1d(np.asarray(lon, dtype=np.float64)),
            order=self.level,
        )

    def cell_boundaries(self, cell_ids, backend="shapely") -> np.ndarray:
        """Cell boundary polygons (``shapely.Polygon``; the one supported backend).

        ``mort2polygon`` returns closed ``[lat, lon]`` rings (4 unique vertices
        plus the repeated first) — and the bare ring (not a one-element list)
        for a single cell, re-wrapped here. The four vertices are recentered
        around the prime meridian before ring construction so dateline-crossing
        and polar cells stay well-formed (mirrors ``xdggs.healpix``'s
        ``cell_boundaries``, which applies the same ``center_around_prime_meridian``).
        """
        if backend != "shapely":
            raise ValueError(f"invalid backend: {backend!r} (only 'shapely' is supported)")
        import shapely
        from mortie import mort2polygon

        rings = mort2polygon(_words(cell_ids))
        if np.asarray(cell_ids).size == 1:
            rings = [rings]
        if len(rings) == 0:
            return shapely.polygons([])
        # Drop the closing vertex -> (n, 4, 2) rings of [lat, lon].
        verts = np.stack([np.asarray(ring, dtype=np.float64)[:-1] for ring in rings])
        lat = verts[..., 0]
        lon = center_around_prime_meridian(verts[..., 1], lat)
        return shapely.polygons(shapely.linearrings(np.stack((lon, lat), axis=-1)))

    def zoom_to(self, cell_ids, level: int) -> np.ndarray:
        """Cells at another order — the xdggs zoom semantics.

        Coarser: one parent per input cell (``clip2order``; length and
        duplicates preserved). Finer: a ``(n, 4**(level - self.level))``
        children array (extra trailing dimension, like the healpix backend).
        Same order: the cells unchanged.
        """
        if level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer between 0 and 29")
        words = _words(cell_ids)
        if level == self.level:
            return words
        from mortie import clip2order, generate_morton_children

        if level < self.level:
            return np.asarray(clip2order(level, words), dtype=np.uint64)
        return np.stack([generate_morton_children(int(w), level) for w in words])


@register_dggs(GRID_NAME)
class MortonIndex(DGGSIndex):
    """``PandasIndex``-backed DGGS index over packed ``uint64`` morton words."""

    _grid: DGGSInfo

    def __init__(self, cell_ids, dim: str, name: str, grid_info: DGGSInfo):
        super().__init__(cell_ids, dim, grid_info)
        self._name = name
        self._index.index.name = name

    @classmethod
    def from_variables(cls, variables, *, options) -> "MortonIndex":
        name, var, dim = _extract_cell_id_variable(variables)
        grid_info = MortonInfo.from_dict(dict(var.attrs) | dict(options or {}))
        return cls(var.data, dim, name, grid_info)

    @property
    def grid_info(self) -> MortonInfo:
        return self._grid

    def _replace(self, new_index) -> "MortonIndex":
        return type(self)(new_index, self._dim, self._name, self._grid)

    def __repr__(self):
        return f"<MortonIndex(level={self._grid.level})>"

    def _repr_inline_(self, max_width: int):
        return f"MortonIndex(level={self._grid.level})"


def decode(ds: xr.Dataset, name: str = "morton", **options) -> xr.Dataset:
    """Assign a :class:`MortonIndex` to the ``name`` coordinate.

    The xdggs-convention decode pattern: build the index off the coord's grid
    attrs (merged with ``**options``, e.g. ``level=...``), assign it via
    ``xr.Coordinates.from_xindex``. Missing attrs are filled in —
    ``grid_name`` is always ``"morton"``, and ``level`` falls back to
    ``ds.attrs["morton_hive"]["cell_order"]`` (the ``open_hive`` manifest
    summary), then to the order packed in the first word itself. The filled
    attrs land on the returned coord, so ``xdggs.decode`` round-trips.
    """
    try:
        var = ds[name].variable
    except KeyError:
        raise ValueError(f"no {name!r} coordinate to index at") from None
    attrs = dict(var.attrs)
    attrs.setdefault("grid_name", GRID_NAME)
    if attrs["grid_name"] != GRID_NAME:
        raise ValueError(f"{name!r} carries grid_name {attrs['grid_name']!r}, not {GRID_NAME!r}")
    if "level" not in attrs and "level" not in options and "cell_order" not in attrs:
        hive = ds.attrs.get("morton_hive") or {}
        if "cell_order" in hive:
            attrs["level"] = int(hive["cell_order"])
        else:
            from mortie import infer_order_from_morton

            attrs["level"] = int(infer_order_from_morton(_words(var.data[:1])))
    var = var.copy(deep=False)
    var.attrs = attrs
    index = MortonIndex.from_variables({name: var}, options=options)
    decoded = ds.assign_coords(xr.Coordinates.from_xindex(index))
    decoded[name].attrs.update(index.grid_info.to_dict())
    return decoded

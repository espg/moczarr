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
direction (``geographic2cell_ids``) does.

Two layers of order discipline follow from that (issue #8):
:class:`MortonInfo` is a transform helper usable on any words — it stays
level-tolerant on the read direction but takes ONE order per call
(:func:`_single_order`); :class:`MortonIndex` binds ``level`` to an actual
coordinate, so it validates once at construction that the domain is
order-uniform *and* at ``level`` (:func:`_check_domain`), which is what makes
"``level`` is a frozen parameter" true of an index.
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

from moczarr.convention import MORTON_CONVENTION_UUID

GRID_NAME = "morton"


def _morton_convention_block(ds: xr.Dataset) -> dict | None:
    """The dataset's §5 morton dggs declaration, or ``None``.

    Recognizes the writer-flip attrs shape (mortie spec §5): an attrs
    ``dggs`` block with ``name: "morton"``, and/or the permanent self-
    declared convention UUID in ``zarr_conventions``. Hard-rejects the
    RETIRED ``name: "healpix"`` + ``indexing_scheme: "morton"`` shape — the
    scheme-blind misread hazard D16 removed: a healpix reader that ignores
    the indexing scheme would decode packed morton words as NESTED ids and
    mis-place every cell, so that shape must never decode quietly.
    """
    block = ds.attrs.get("dggs")
    if not isinstance(block, dict):
        return None
    if block.get("name") == "healpix" and block.get("indexing_scheme") == "morton":
        raise ValueError(
            "dataset declares dggs name 'healpix' with indexing_scheme 'morton' "
            "— the retired pre-flip shape (mortie spec §5). A scheme-blind "
            "healpix reader would silently decode morton words as NESTED ids "
            "and mis-place every cell; morton-declared stores use name "
            "'morton'. Regenerate the store with a post-flip (zagg#314) writer."
        )
    conventions = ds.attrs.get("zarr_conventions")
    declared_by_uuid = isinstance(conventions, list) and any(
        isinstance(entry, dict) and entry.get("uuid") == MORTON_CONVENTION_UUID
        for entry in conventions
    )
    if block.get("name") == GRID_NAME or declared_by_uuid:
        return dict(block)
    return None


def _words(cell_ids) -> np.ndarray:
    """Cell ids as a contiguous ``uint64`` array (mortie's kernel input)."""
    return np.ascontiguousarray(cell_ids, dtype=np.uint64)


def _single_order(words: np.ndarray, surface: str) -> int | None:
    """The one order ``words`` share — ``None`` for an empty array.

    The :class:`MortonInfo` cell-reading contract: ONE order per call. This is
    a requirement of the surface, NOT a claim about ``level`` — the words are
    self-describing and ``level`` is not consulted on this direction (see the
    module docstring), so a uniform off-``level`` array is fine while a
    mixed-order one is ambiguous *here* whatever ``level`` says. mortie 0.9.1+
    converts mixed orders natively (mortie#116), so nothing below is an
    upstream limitation: it is that these surfaces have one order-keyed answer
    to give (one lon/lat per cell at its own order is fine; one trailing
    children dimension, or one boundary set, is not).

    The domain-wide version of the same idea is :func:`_check_domain` — an
    *index* pins its order to ``grid_info.level`` once at construction, so
    index-backed calls satisfy this for free.
    """
    from mortie import orders_of

    if words.size == 0:
        return None
    distinct = np.unique(orders_of(words))
    if distinct.size > 1:
        raise ValueError(
            f"mixed-order cell_ids (orders {[int(o) for o in distinct]}): "
            f"MortonInfo.{surface} reads one order per call. Split by mortie.orders_of "
            f"and call per order — a pyramid/overview coordinate (zagg#262) is a stack "
            f"of grids, not one."
        )
    return int(distinct[0])


def _check_domain(words: np.ndarray, level: int) -> None:
    """The :class:`MortonIndex` domain contract: ONE order, and it is ``level``.

    Validated once at construction — the single place ``level`` is bound to
    data — so every read surface (``cell_centers``, ``cell_boundaries``,
    ``zoom_to``, ``sel_latlon``) inherits it instead of each guarding
    separately (PR #22 Question 2; issue #8). Both halves are load-bearing:

    * one order, because an xdggs index is single-level by construction and a
      mixed-order coordinate is a stack of grids, not one;
    * *that* order, because ``geographic2cell_ids`` bins at ``level`` and
      ``sel_latlon`` then looks the result up in this coordinate — an
      off-``level`` domain matches nothing, silently.

    ``MortonRanges.from_cell_words`` already enforced exactly this for the
    ``"moc"`` kind; the ``"pandas"`` kind was the laxer half.
    """
    from mortie import orders_of

    if words.size == 0:
        return
    distinct = np.unique(orders_of(words))
    if distinct.size > 1:
        raise ValueError(
            f"mixed-order morton domain (orders {[int(o) for o in distinct]}): a "
            f"MortonIndex carries ONE order — an xdggs index is single-level by "
            f"construction (level is a frozen MortonInfo parameter). Decode each level "
            f"separately (split by mortie.orders_of); a pyramid/overview coordinate "
            f"(zagg#262) is a stack of grids, not one index."
        )
    order = int(distinct[0])
    if order != level:
        raise ValueError(
            f"morton domain is at order {order}, not the declared level {level}: an "
            f"index binds level to its coordinate (geographic2cell_ids bins at level and "
            f"sel_latlon looks the result up here, so an off-level domain matches "
            f"nothing). Pass level={order}, or omit level to read it from the words."
        )


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

        ``mort2geo`` reads each word's own order — ``level`` is not consulted —
        and returns lon in [0, 360); the xdggs surface is [-180, 180).

        Mixed-order ``cell_ids`` reject here as this *surface's* one-order-
        per-call contract, not as a kernel limitation and not as a claim about
        ``level`` (issue #8): mortie 0.9.1+ converts mixed orders natively
        (mortie#116), and this method deliberately ignores ``level``, so
        citing a "frozen level" would describe an invariant it does not
        enforce. The domain-wide contract (one order, *and* that order is
        ``level``) is enforced once in :class:`MortonIndex` instead — see
        :func:`_check_domain`. Decode each level separately.
        """
        from mortie import mort2geo

        words = _words(cell_ids)
        _single_order(words, "cell_ids2geographic")
        lat, lon = mort2geo(words)
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
        """Cell boundary polygons — ``backend="shapely"`` (default) or ``"geoarrow"``.

        Mirrors ``xdggs.healpix``'s ``cell_boundaries`` contract by dispatching
        to its backend builders: ``"shapely"`` returns an array of
        :py:class:`shapely.Polygon` (what the accessor consumes);
        ``"geoarrow"`` returns a geoarrow polygon array tagged with spherical
        edges (the lonboard-fast path). ``mort2polygon`` returns closed
        ``[lat, lon]`` rings (4 unique vertices plus the repeated first) — and
        the bare ring (not a one-element list) for a single cell, re-wrapped
        here. The four vertices are recentered around the prime meridian
        before ring construction so dateline-crossing and polar cells stay
        well-formed (same ``center_around_prime_meridian`` xdggs applies).

        Mixed-order input rejects, same one-order-per-call contract as
        :meth:`cell_ids2geographic` (issue #8 fold): ``mort2polygon`` would
        happily emit rings at several orders, but a boundary set that silently
        mixes cell sizes is the bypass that let a mixed-order coordinate reach
        the accessor at all.
        """
        from mortie import mort2polygon
        from xdggs.healpix import polygons_geoarrow, polygons_shapely

        backends = {"shapely": polygons_shapely, "geoarrow": polygons_geoarrow}
        backend_func = backends.get(backend)
        if backend_func is None:
            raise ValueError(f"invalid backend: {backend!r} (one of {sorted(backends)})")
        words = _words(cell_ids)
        _single_order(words, "cell_boundaries")
        rings = mort2polygon(words)
        if np.asarray(cell_ids).size == 1:
            rings = [rings]
        if len(rings) == 0:
            if backend == "geoarrow":
                # arro3's list_array rejects 0-length dimensions; upstream
                # xdggs shares the limitation, so fail pointedly.
                raise ValueError("backend='geoarrow' does not support zero cells (arro3 limit)")
            import shapely

            return shapely.polygons(shapely.linearrings(np.empty((0, 4, 2))))
        # Drop the closing vertex -> (n, 4, 2) rings of [lat, lon].
        verts = np.stack([np.asarray(ring, dtype=np.float64)[:-1] for ring in rings])
        lat = verts[..., 0]
        lon = center_around_prime_meridian(verts[..., 1], lat)
        return backend_func(np.stack((lon, lat), axis=-1))

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
        # Finer: one ``generate_morton_children`` call per parent — an O(n)
        # Python loop. Revisited at the mortie#116 adoption (issue #8):
        # 0.9.1's group-by-order dispatch landed in the geo kernels only —
        # there is still no vectorized many-parent children kernel
        # (``split_children`` builds a compacted trie, different semantics).
        # Empty input mirrors the coarser ``(0,)`` with a clean ``(0, 4**diff)``
        # rather than letting ``np.stack`` raise on an empty sequence.
        if words.size == 0:
            return np.empty((0, 4 ** (level - self.level)), dtype=np.uint64)
        return np.stack([generate_morton_children(int(w), level) for w in words])


@register_dggs(GRID_NAME)
class MortonIndex(DGGSIndex):
    """DGGS index over packed ``uint64`` morton words — pandas- or MOC-backed.

    ``index_kind`` mirrors upstream ``HealpixIndex``: ``"pandas"`` (default)
    materializes a ``PandasIndex``; ``"moc"`` wraps the core
    :class:`moczarr.moc_index.MortonMocIndex` (interval-set domain, fabricated
    coordinate) so the ``ds.dggs`` accessor works on a lazy index. An
    ``xr.Index`` instance passes through as the inner index directly (the
    ``open_hive(index_kind="moc", decode=True)`` wrap — no rebuild).

    The ``"pandas"`` default here is the decode-a-materialized-coordinate
    convention (upstream parity); :func:`moczarr.open_hive` itself defaults
    to ``index_kind="moc"`` and passes the kind through explicitly.

    The domain contract — one order, and that order is ``grid_info.level`` —
    is checked HERE, once, whenever the words are handed in raw
    (:func:`_check_domain`; issue #8, PR #22 Question 2). That covers every
    read surface at once rather than guarding them one at a time, and it makes
    "``level`` is a frozen parameter" literally true of an index. A passed
    ``xr.Index`` is trusted for its words (an already-built index, or the
    ``_replace`` a ``sel``/``isel`` makes from a validated one — revalidating
    there would put an ``O(n)`` scan on every selection); a
    :class:`~moczarr.moc_index.MortonMocIndex` still gets its ``O(1)``
    ``cell_order`` cross-checked.
    """

    _grid: DGGSInfo

    def __init__(
        self, cell_ids, dim: str, name: str, grid_info: DGGSInfo, index_kind: str = "pandas"
    ):
        if index_kind not in ("pandas", "moc"):
            raise ValueError(f"index_kind={index_kind!r}: expected 'pandas' or 'moc'")
        self._dim = dim
        self._name = name
        if isinstance(cell_ids, xr.Index):
            from moczarr.moc_index import MortonMocIndex

            if isinstance(cell_ids, MortonMocIndex) and cell_ids.level != grid_info.level:
                raise ValueError(
                    f"morton domain is at order {cell_ids.level}, not the declared level "
                    f"{grid_info.level}: an index binds level to its coordinate. Pass "
                    f"level={cell_ids.level}, or omit level to read it from the store."
                )
            self._index = cell_ids
        elif index_kind == "pandas":
            from xarray.indexes import PandasIndex

            _check_domain(_words(cell_ids), grid_info.level)
            self._index = PandasIndex(cell_ids, dim)
            self._index.index.name = name
        else:
            from moczarr.moc_index import MortonMocIndex
            from moczarr.ranges import MortonRanges

            words = _words(cell_ids)
            _check_domain(words, grid_info.level)
            ranges = MortonRanges.from_cell_words(words, grid_info.level)
            self._index = MortonMocIndex(ranges, dim=dim, name=name)
        self._kind = index_kind
        self._grid = grid_info

    def values(self):
        if self._kind == "moc":
            return self._index.ranges.fabricate()
        return self._index.index.values

    @classmethod
    def from_variables(cls, variables, *, options) -> "MortonIndex":
        name, var, dim = _extract_cell_id_variable(variables)
        options = dict(options or {})
        index_kind = options.pop("index_kind", "pandas")
        grid_info = MortonInfo.from_dict(dict(var.attrs) | options)
        return cls(var.data, dim, name, grid_info, index_kind=index_kind)

    @property
    def grid_info(self) -> MortonInfo:
        return self._grid

    def _replace(self, new_index) -> "MortonIndex":
        return type(self)(new_index, self._dim, self._name, self._grid, index_kind=self._kind)

    def __repr__(self):
        return f"<MortonIndex(level={self._grid.level}, kind={self._kind})>"

    def _repr_inline_(self, max_width: int):
        return f"MortonIndex(level={self._grid.level}, kind={self._kind})"


def decode(
    ds: xr.Dataset, name: str = "morton", index_kind: str = "pandas", **options
) -> xr.Dataset:
    """Assign a :class:`MortonIndex` to the ``name`` coordinate.

    The xdggs-convention decode pattern: build the index off the coord's grid
    attrs (merged with ``**options``, e.g. ``level=...``), assign it via
    ``xr.Coordinates.from_xindex``. Missing attrs are filled in —
    ``grid_name`` is always ``"morton"``, and ``level`` falls back to the §5
    convention block's ``refinement_level`` (a dataset carrying the
    writer-flip ``dggs``/``zarr_conventions`` attrs — mortie spec §5,
    recognized by ``name: "morton"`` and/or the permanent convention UUID),
    then to ``ds.attrs["morton_hive"]["cell_order"]`` (the ``open_hive``
    manifest summary), then to the order packed in the first word itself.
    The retired ``name: "healpix"`` + ``indexing_scheme: "morton"`` attrs
    shape hard-rejects with a diagnostic (§5: the scheme-blind misread
    hazard). The filled attrs land on the returned coord, so
    ``xdggs.decode`` round-trips. xdggs's own zarr-convention path
    (``xdggs.decode(ds, convention="zarr", name="morton")``) reaches the
    same registered grid directly off the §5 block.

    ``index_kind="moc"`` backs the index with the lazy
    :class:`~moczarr.moc_index.MortonMocIndex`: a coordinate already indexed
    by one (the ``open_hive(index_kind="moc")`` result) is wrapped as-is;
    otherwise the materialized words run-detect into intervals.
    """
    try:
        var = ds[name].variable
    except KeyError:
        raise ValueError(f"no {name!r} coordinate to index at") from None
    attrs = dict(var.attrs)
    attrs.setdefault("grid_name", GRID_NAME)
    if attrs["grid_name"] != GRID_NAME:
        raise ValueError(f"{name!r} carries grid_name {attrs['grid_name']!r}, not {GRID_NAME!r}")
    # The §5 convention block (writer-flip attrs shape): recognized whenever
    # present — and the retired healpix+morton shape hard-rejects here even
    # when the level comes from elsewhere.
    block = _morton_convention_block(ds)
    if "level" not in attrs and "level" not in options and "cell_order" not in attrs:
        hive = ds.attrs.get("morton_hive") or {}
        if block is not None and "refinement_level" in block:
            attrs["level"] = int(block["refinement_level"])
        elif "cell_order" in hive:
            attrs["level"] = int(hive["cell_order"])
        else:
            from mortie import infer_order_from_morton

            attrs["level"] = int(infer_order_from_morton(_words(var.data[:1])))
    var = var.copy(deep=False)
    var.attrs = attrs
    existing = ds.xindexes.get(name) if index_kind == "moc" else None
    if existing is not None:
        from moczarr.moc_index import MortonMocIndex

        if isinstance(existing, MortonMocIndex):
            grid_info = MortonInfo.from_dict(attrs | dict(options))
            index: MortonIndex = MortonIndex(
                existing, ds[name].dims[0], name, grid_info, index_kind="moc"
            )
        else:
            existing = None
    if existing is None:
        index = MortonIndex.from_variables(
            {name: var}, options=dict(options) | {"index_kind": index_kind}
        )
    decoded = ds.assign_coords(xr.Coordinates.from_xindex(index))
    decoded[name].attrs.update(index.grid_info.to_dict())
    return decoded

"""``moczarr.dggs`` — the xdggs ``"morton"`` grid, against goldens + the SERC fixture.

Skipped wholesale in core-only envs: everything below needs the
``moczarr[xdggs]`` extra (xdggs + shapely), and ``moczarr.dggs`` itself
raises the pointed install hint when xdggs is absent.
"""

from pathlib import Path

import numpy as np
import pytest

from moczarr import convention, open_hive, store

xdggs = pytest.importorskip("xdggs")
shapely = pytest.importorskip("shapely")
dggs = pytest.importorskip("moczarr.dggs")

FIXTURE = Path(__file__).parent / "data" / "serc_hive"
SERC_SHARD = "4331422"
#: The order-6 golden shard family (southern base -5) from the zagg goldens.
GOLDEN = "-5112333"


@pytest.fixture()
def serc():
    return str(FIXTURE)


def _decimals(words):
    return [convention.morton_decimal(int(w)) for w in np.ravel(words)]


def _golden_family():
    """The four order-6 siblings of the golden shard, ascending."""
    return np.asarray([convention.morton_word(GOLDEN[:-1] + d) for d in "1234"], dtype=np.uint64)


def _occupied_cell(serc):
    """The one occupied (bitmap-listed) order-8 cell of the SERC shard."""
    occupied = store.read_coverage_bitmap(serc, convention.leaf_path(SERC_SHARD))
    return np.uint64(occupied[0])


class TestMortonInfo:
    def test_dict_roundtrip(self):
        info = dggs.MortonInfo.from_dict({"grid_name": "morton", "level": 8})
        assert info.to_dict() == {"grid_name": "morton", "level": 8}
        assert dggs.MortonInfo.from_dict(info.to_dict()) == info

    def test_from_dict_accepts_cell_order(self):
        # The manifest spells level "cell_order"; from_dict translates it.
        assert dggs.MortonInfo.from_dict({"grid_name": "morton", "cell_order": 8}).level == 8

    def test_level_bounds(self):
        assert dggs.MortonInfo(level=0).level == 0
        assert dggs.MortonInfo(level=29).level == 29
        for bad in (-1, 30):
            with pytest.raises(ValueError, match="level"):
                dggs.MortonInfo(level=bad)

    def test_geographic_roundtrip(self):
        info = dggs.MortonInfo(level=6)
        words = _golden_family()
        lon, lat = info.cell_ids2geographic(words)
        assert ((lon >= -180.0) & (lon < 180.0)).all()
        np.testing.assert_array_equal(info.geographic2cell_ids(lon, lat), words)

    def test_geographic_single_cell(self):
        # mort2geo collapses 1-element input to scalars; the surface stays 1-d.
        info = dggs.MortonInfo(level=6)
        word = np.asarray([convention.morton_word(GOLDEN)], dtype=np.uint64)
        lon, lat = info.cell_ids2geographic(word)
        assert lon.shape == (1,) and lat.shape == (1,)
        np.testing.assert_array_equal(info.geographic2cell_ids(lon, lat), word)

    def test_geographic_ignores_level_but_rejects_mixed_orders(self):
        # Words carry their own depth: the info's level is not consulted on
        # the ids->geo direction, but mortie rejects mixed orders per call.
        info = dggs.MortonInfo(level=3)
        words = _golden_family()  # order 6, deliberately not the info's level
        lon, lat = info.cell_ids2geographic(words)
        np.testing.assert_array_equal(dggs.MortonInfo(level=6).geographic2cell_ids(lon, lat), words)
        mixed = np.asarray(
            [convention.morton_word(GOLDEN), convention.morton_word(GOLDEN + "11")],
            dtype=np.uint64,
        )
        with pytest.raises(ValueError, match="[Mm]ixed orders"):
            info.cell_ids2geographic(mixed)

    def test_cell_boundaries_centroids(self):
        info = dggs.MortonInfo(level=6)
        words = _golden_family()
        polys = info.cell_boundaries(words)
        lon, lat = info.cell_ids2geographic(words)
        assert polys.shape == words.shape
        for poly, x, y in zip(polys, lon, lat):
            assert poly.geom_type == "Polygon"
            assert abs(poly.centroid.x - x) < 0.3 and abs(poly.centroid.y - y) < 0.3

    def test_cell_boundaries_single_cell(self):
        # mort2polygon hands back the bare ring for one cell; still one Polygon.
        info = dggs.MortonInfo(level=6)
        polys = info.cell_boundaries(np.asarray([convention.morton_word(GOLDEN)], dtype=np.uint64))
        assert polys.shape == (1,) and polys[0].geom_type == "Polygon"

    def test_cell_boundaries_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="backend"):
            dggs.MortonInfo(level=6).cell_boundaries(_golden_family(), backend="geoarrow")

    def test_zoom_to_coarser_is_decimal_truncation(self):
        # The §5 predicate: the parent decimal is the child decimal minus its
        # last digit. Length (and duplicates) preserved, one parent per input.
        info = dggs.MortonInfo(level=6)
        words = _golden_family()
        parents = info.zoom_to(words, 5)
        assert parents.shape == words.shape
        assert _decimals(parents) == [d[:-1] for d in _decimals(words)]
        assert len(set(_decimals(parents))) == 1  # four siblings, one parent

    def test_zoom_to_finer_children_prefix(self):
        info = dggs.MortonInfo(level=6)
        words = _golden_family()[:2]
        children = info.zoom_to(words, 8)
        assert children.shape == (2, 16) and children.dtype == np.uint64
        for row, parent in zip(children, _decimals(words)):
            assert all(d.startswith(parent) for d in _decimals(row))

    def test_zoom_to_same_level(self):
        words = _golden_family()
        np.testing.assert_array_equal(dggs.MortonInfo(level=6).zoom_to(words, 6), words)

    def test_zoom_to_rejects_bad_level(self):
        with pytest.raises(ValueError, match="level"):
            dggs.MortonInfo(level=6).zoom_to(_golden_family(), 30)


class TestMortonIndex:
    def test_registered_in_public_registry(self):
        from xdggs.utils import GRID_REGISTRY

        assert GRID_REGISTRY["morton"] is dggs.MortonIndex

    def test_open_hive_decode(self, serc):
        ds = open_hive(serc, decode=True)
        assert ds.dggs.grid_info == dggs.MortonInfo(level=8)

    def test_default_is_undecoded(self, serc):
        assert "morton" not in open_hive(serc).xindexes

    def test_single_index_morton_only(self, serc):
        # One DGGSIndex per dataset (the accessor's rule): morton is indexed,
        # cell_ids stays a plain coordinate.
        ds = open_hive(serc, decode=True)
        assert set(ds.xindexes) == {"morton"}
        index = ds.xindexes["morton"]
        assert isinstance(index, dggs.MortonIndex)
        assert "cell_ids" in ds.coords and "cell_ids" not in ds.xindexes
        assert repr(index) == "<MortonIndex(level=8)>"
        assert index._repr_inline_(80) == "MortonIndex(level=8)"

    def test_isel_keeps_index(self, serc):
        ds = open_hive(serc, decode=True).isel(cells=[0, 1])
        assert isinstance(ds.xindexes["morton"], dggs.MortonIndex)

    def test_sel_word(self, serc):
        ds = open_hive(serc, decode=True)
        word = _occupied_cell(serc)
        row = ds.sel(morton=word)
        assert np.uint64(row["morton"].values) == word
        assert int(row["count"].values) > 0  # the occupied cell has data

    def test_sel_latlon(self, serc):
        ds = open_hive(serc, decode=True)
        word = _occupied_cell(serc)
        lon, lat = ds.dggs.grid_info.cell_ids2geographic(np.asarray([word], dtype=np.uint64))
        sub = ds.dggs.sel_latlon(lat, lon)
        np.testing.assert_array_equal(np.atleast_1d(sub["morton"].values).astype(np.uint64), [word])
        assert int(np.atleast_1d(sub["count"].values)[0]) == int(ds.sel(morton=word)["count"])

    def test_cell_boundaries_cover_centers(self, serc):
        ds = open_hive(serc, decode=True)
        boundaries = ds.dggs.cell_boundaries()
        assert boundaries.shape == (ds.sizes["cells"],)
        lon, lat = ds.dggs.grid_info.cell_ids2geographic(
            np.asarray(ds["morton"].values, dtype=np.uint64)
        )
        assert all(
            poly.covers(shapely.Point(x, y)) for poly, x, y in zip(boundaries.values, lon, lat)
        )

    def test_cell_centers_roundtrip(self, serc):
        ds = open_hive(serc, decode=True)
        centers = ds.dggs.cell_centers()
        binned = ds.dggs.grid_info.geographic2cell_ids(
            centers["longitude"].values, centers["latitude"].values
        )
        np.testing.assert_array_equal(binned, np.asarray(ds["morton"].values, dtype=np.uint64))

    def test_zoom_to_accessor(self, serc):
        ds = open_hive(serc, decode=True)
        shards = ds.dggs.zoom_to(6)
        assert shards.dims == ("cells",)
        assert _decimals(shards.values) == [d[:7] for d in _decimals(ds["morton"].values)]
        children = ds.dggs.zoom_to(9)
        assert children.dims == ("cells", "children")
        assert children.shape == (ds.sizes["cells"], 4)


class TestDecode:
    def test_level_from_manifest(self, serc):
        decoded = dggs.decode(open_hive(serc))
        assert decoded.dggs.grid_info.level == 8
        assert decoded["morton"].attrs == {"grid_name": "morton", "level": 8}

    def test_level_inferred_from_word(self, serc):
        # No manifest summary: the packed word itself carries its depth.
        ds = open_hive(serc)
        del ds.attrs["morton_hive"]
        assert dggs.decode(ds).dggs.grid_info.level == 8

    def test_level_option_wins(self, serc):
        assert dggs.decode(open_hive(serc), level=7).dggs.grid_info.level == 7

    def test_wrong_grid_name_raises(self, serc):
        ds = open_hive(serc)
        ds["morton"].attrs["grid_name"] = "healpix"
        with pytest.raises(ValueError, match="grid_name"):
            dggs.decode(ds)

    def test_missing_coord_raises(self, serc):
        with pytest.raises(ValueError, match="'nope'"):
            dggs.decode(open_hive(serc), name="nope")

    def test_upstream_xdggs_decode(self, serc):
        # The registry claim end-to-end: plain xdggs.decode resolves
        # grid_name "morton" to MortonIndex off the coord attrs alone.
        ds = open_hive(serc)
        ds["morton"].attrs.update({"grid_name": "morton", "level": 8})
        decoded = xdggs.decode(ds, name="morton")
        assert isinstance(decoded.xindexes["morton"], dggs.MortonIndex)
        assert decoded.dggs.grid_info == dggs.MortonInfo(level=8)

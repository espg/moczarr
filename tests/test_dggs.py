"""``moczarr.dggs`` — the xdggs ``"morton"`` grid, against goldens + the SERC fixture.

Skipped wholesale in core-only envs: everything below needs the
``moczarr[xdggs]`` extra (xdggs + shapely), and ``moczarr.dggs`` itself
raises the pointed install hint when xdggs is absent.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from test_fabricate import _dual_written_copy, golden_cell_ids

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

    def test_cell_boundaries_dateline(self):
        # A cell straddling ±180° must recenter around the prime meridian, else
        # the raw mortie ring wraps the long way (~7× area, center uncovered).
        info = dggs.MortonInfo(level=6)
        dateline = np.ascontiguousarray(
            info.geographic2cell_ids(np.array([180.0]), np.array([0.0])), dtype=np.uint64
        )
        non_dateline = np.ascontiguousarray(
            info.geographic2cell_ids(np.array([0.0]), np.array([0.0])), dtype=np.uint64
        )
        poly = info.cell_boundaries(dateline)[0]
        ref = info.cell_boundaries(non_dateline)[0]
        lon, lat = info.cell_ids2geographic(dateline)
        # center normalizes to −180 ≡ +180; the recentered ring sits at +180.
        assert poly.covers(shapely.Point(lon[0] % 360.0, lat[0]))
        assert poly.area < 2.0 * ref.area

    def test_cell_boundaries_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="invalid backend"):
            dggs.MortonInfo(level=6).cell_boundaries(_golden_family(), backend="wkb")

    def test_cell_boundaries_geoarrow(self):
        # The lonboard-fast path: a geoarrow polygon array with spherical-edge
        # extension metadata, one geometry per cell (xdggs backend contract).
        info = dggs.MortonInfo(level=6)
        words = _golden_family()
        arr = info.cell_boundaries(words, backend="geoarrow")
        assert len(arr) == words.size
        metadata = arr.field.metadata
        assert metadata[b"ARROW:extension:name"] == b"geoarrow.polygon"
        assert b'"edges": "spherical"' in metadata[b"ARROW:extension:metadata"]

    def test_cell_boundaries_geoarrow_rejects_empty(self):
        # arro3 cannot build 0-length list arrays; fail pointedly rather than
        # letting the backend crash opaquely (shapely returns an empty array).
        info = dggs.MortonInfo(level=6)
        empty = np.asarray([], dtype=np.uint64)
        assert info.cell_boundaries(empty).size == 0
        with pytest.raises(ValueError, match="zero cells"):
            info.cell_boundaries(empty, backend="geoarrow")

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

    def test_zoom_to_finer_empty(self):
        # A zero-cell selection zooms cleanly to (0, 4**diff), not np.stack raising.
        empty = np.asarray([], dtype=np.uint64)
        children = dggs.MortonInfo(level=6).zoom_to(empty, 8)
        assert children.shape == (0, 16) and children.dtype == np.uint64


class TestMortonIndex:
    def test_registered_in_public_registry(self):
        from xdggs.utils import GRID_REGISTRY

        assert GRID_REGISTRY["morton"] is dggs.MortonIndex

    def test_open_hive_decode(self, serc):
        ds = open_hive(serc, decode=True)
        assert ds.dggs.grid_info == dggs.MortonInfo(level=8)

    def test_default_is_undecoded(self, serc):
        # The default open carries the core lazy index (the phase-7d moc
        # flip, issue #1) but stays undecoded: no xdggs MortonIndex — and so
        # no ds.dggs accessor surface — until decode=True.
        from moczarr.moc_index import MortonMocIndex

        index = open_hive(serc).xindexes["morton"]
        assert isinstance(index, MortonMocIndex)
        assert not isinstance(index, dggs.MortonIndex)

    def test_single_index_morton_only(self, serc):
        # One DGGSIndex per dataset (the accessor's rule): morton is indexed,
        # cell_ids stays a plain coordinate. The default decode wraps the
        # lazy moc index (the phase-7d flip).
        ds = open_hive(serc, decode=True)
        assert set(ds.xindexes) == {"morton"}
        index = ds.xindexes["morton"]
        assert isinstance(index, dggs.MortonIndex)
        assert "cell_ids" in ds.coords and "cell_ids" not in ds.xindexes
        assert repr(index) == "<MortonIndex(level=8, kind=moc)>"
        assert index._repr_inline_(80) == "MortonIndex(level=8, kind=moc)"

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


class TestCombinedFlags:
    """Both #6/#7 knobs together: fabrication feeds decode (the merge-order test)."""

    def test_decode_and_fabricate_morton_only(self, serc):
        # Morton-only store (the fixture, post zagg#314), both flags on:
        # fabrication attaches cell_ids BEFORE decode indexes morton, and the
        # fabricated coordinate is byte-equal to the frozen dual-written golden.
        ds = open_hive(serc, fabricate_cell_ids="auto", decode=True)
        assert isinstance(ds.xindexes["morton"], dggs.MortonIndex)
        assert ds.dggs.grid_info == dggs.MortonInfo(level=8)
        assert "cell_ids" in ds.coords and "cell_ids" not in ds.xindexes
        assert ds["cell_ids"].dtype == np.uint64
        np.testing.assert_array_equal(ds["cell_ids"].values, golden_cell_ids())

    def test_decode_and_fabricate_dual_written(self, tmp_path):
        # Dual-written (emit_cell_ids hatch) shape with both flags: the
        # stored cell_ids rides through untouched (deviant bytes kept) and
        # morton still gets its index.
        ds = open_hive(
            _dual_written_copy(tmp_path), fabricate_cell_ids="auto", decode=True, index_kind="pandas"
        )
        assert isinstance(ds.xindexes["morton"], dggs.MortonIndex)
        assert ds.dggs.grid_info == dggs.MortonInfo(level=8)
        np.testing.assert_array_equal(ds["cell_ids"].values, golden_cell_ids() + np.uint64(1))


class TestConventionBlock:
    """The mortie spec §5 zarr DGGS convention block (writer-flip attrs)."""

    def _writer_flip_attrs(self, level):
        # Hand-built EXACTLY per spec §5: the dggs declaration (name and
        # coordinate both "morton", no kind/resolution field — §4) plus the
        # self-declared zarr_conventions entry.
        return {
            "zarr_conventions": [convention.MORTON_CONVENTION_ENTRY],
            "dggs": {
                "name": "morton",
                "coordinate": "morton",
                "refinement_level": level,
                "spatial_dimension": "cells",
                "compression": "none",
            },
        }

    def test_constants_match_spec(self):
        assert convention.MORTON_CONVENTION_UUID == "3e22156d-ea9e-4e01-95fe-e3809a4b41e7"
        entry = convention.MORTON_CONVENTION_ENTRY
        assert entry["uuid"] == convention.MORTON_CONVENTION_UUID
        assert entry["name"] == "morton-dggs"
        assert entry["schema_url"].endswith("docs/specification.md#dggs-attrs")
        assert entry["spec_url"].endswith("espg/mortie/blob/main/docs/specification.md")

    def test_decode_reads_block_level(self):
        # No coord attrs, no morton_hive summary: level comes from the §5
        # block's refinement_level (grid_name/level translation).
        ds = xr.Dataset(coords={"morton": ("cells", _golden_family())})
        ds.attrs.update(self._writer_flip_attrs(level=6))
        out = dggs.decode(ds)
        assert isinstance(out.xindexes["morton"], dggs.MortonIndex)
        assert out.dggs.grid_info == dggs.MortonInfo(level=6)

    def test_decode_recognizes_uuid_only(self):
        # Recognition also keys on the permanent UUID in zarr_conventions
        # (readers may key on it — §5), independent of the block's name.
        ds = xr.Dataset(coords={"morton": ("cells", _golden_family())})
        attrs = self._writer_flip_attrs(level=6)
        del attrs["dggs"]["name"]
        ds.attrs.update(attrs)
        out = dggs.decode(ds)
        assert out.dggs.grid_info == dggs.MortonInfo(level=6)

    def test_real_fixture_attrs_carry_the_block(self, serc):
        # The regenerated (post zagg#314) fixture IS writer-flip shaped: the
        # opened dataset carries the §5 attrs and decode reads them.
        ds = open_hive(serc)
        block = ds.attrs["dggs"]
        assert block["name"] == "morton" and block["coordinate"] == "morton"
        entries = {e["uuid"]: e for e in ds.attrs["zarr_conventions"]}
        assert entries[convention.MORTON_CONVENTION_UUID] == convention.MORTON_CONVENTION_ENTRY
        del ds.attrs["morton_hive"]  # force the §5-block level path
        out = dggs.decode(ds)
        assert out.dggs.grid_info == dggs.MortonInfo(level=8)

    def test_retired_healpix_morton_shape_hard_rejects(self):
        # §5: name "healpix" + indexing_scheme "morton" is the scheme-blind
        # misread hazard — decode must refuse with a diagnostic, never parse.
        ds = xr.Dataset(coords={"morton": ("cells", _golden_family())})
        ds.attrs["dggs"] = {
            "name": "healpix",
            "indexing_scheme": "morton",
            "refinement_level": 6,
            "coordinate": "cell_ids",
        }
        with pytest.raises(ValueError, match="scheme-blind"):
            dggs.decode(ds)

    def test_xdggs_zarr_convention_reaches_morton(self, serc):
        # xdggs's own decode(convention="zarr") hits GRID_REGISTRY["morton"]
        # off the §5 block (name= must be given: xdggs 0.6.0's zarr decoder
        # keys the new index on the passed name, and None breaks xarray).
        rel = convention.leaf_path(SERC_SHARD)
        ds = xr.open_zarr(f"{serc}/{rel}", group="8", consolidated=False, zarr_format=3)
        out = xdggs.decode(ds, convention="zarr", name="morton")
        idx = out.xindexes["morton"]
        assert isinstance(idx, dggs.MortonIndex)
        assert idx.grid_info.level == 8
        # The decoder consumes the declaration: block popped, our
        # self-declared entry left in place (it removed only the generic one).
        assert "dggs" not in out.attrs
        uuids = [e["uuid"] for e in out.attrs.get("zarr_conventions", [])]
        assert uuids == [convention.MORTON_CONVENTION_UUID]

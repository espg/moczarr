"""``MortonMocIndex`` — construction, xarray flows, and pandas parity (5b).

The parity classes pin the load-bearing claim: on identical domains, the
lazy index answers ``sel``/``isel``/alignment with the SAME positional
results as the pandas-backed ``MortonIndex`` (the xdggs layer), so flipping
``index_kind`` never changes answers — only what gets materialized. Those
classes skip in core-only envs (they need the ``[xdggs]`` extra to build the
pandas side); everything else is xarray-only.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moczarr import convention, open_hive, store
from moczarr.moc_index import MortonMocIndex, _normalize_chunks
from moczarr.ranges import MortonRanges

FIXTURE = Path(__file__).parent / "data" / "serc_hive"
SERC_SHARD = "4331422"
CELL_ORDER = 8


@pytest.fixture()
def serc():
    return str(FIXTURE)


@pytest.fixture()
def envelope(serc):
    return store.load_root_coverage(serc)


@pytest.fixture()
def moc_index(envelope):
    return MortonMocIndex.from_moc(envelope, cell_order=CELL_ORDER, dim="cells")


@pytest.fixture()
def ds_moc(serc, moc_index):
    ds = open_hive(serc)
    return ds.assign_coords(xr.Coordinates.from_xindex(moc_index))


@pytest.fixture()
def ds_pandas(serc):
    dggs = pytest.importorskip("moczarr.dggs")
    return dggs.decode(open_hive(serc))


class TestConstruction:
    def test_from_moc_sources_agree(self, envelope, moc_index):
        from moczarr.coverage import ranges_words

        by_shards = MortonMocIndex.from_moc(
            ranges_words(envelope), cell_order=CELL_ORDER, dim="cells"
        )
        assert moc_index.equals(by_shards)
        by_ranges = MortonMocIndex.from_moc(
            MortonRanges.from_root_coverage(envelope, CELL_ORDER),
            cell_order=CELL_ORDER,
            dim="cells",
        )
        assert moc_index.equals(by_ranges)

    def test_from_moc_aoi(self, envelope, moc_index):
        word = convention.morton_word(SERC_SHARD)
        aoi_index = MortonMocIndex.from_moc(
            envelope, cell_order=CELL_ORDER, aoi=[word], dim="cells"
        )
        assert aoi_index.size == 16
        np.testing.assert_array_equal(
            aoi_index.ranges.fabricate(), moc_index.ranges.intersect([word]).fabricate()
        )

    def test_from_moc_order_mismatch_raises(self, envelope):
        ranges = MortonRanges.from_root_coverage(envelope, CELL_ORDER)
        with pytest.raises(ValueError, match="do not match cell_order"):
            MortonMocIndex.from_moc(ranges, cell_order=9, dim="cells")

    def test_set_xindex_round_trip(self, serc, moc_index):
        # The materialized->lazy direction: run-detecting the eager opener's
        # coordinate rebuilds the same domain.
        ds = open_hive(serc)
        lazy = ds.set_xindex("morton", MortonMocIndex, level=CELL_ORDER)
        assert lazy.xindexes["morton"].equals(moc_index)
        inferred = ds.set_xindex("morton", MortonMocIndex)
        assert inferred.xindexes["morton"].equals(moc_index)

    def test_set_xindex_unknown_option_raises(self, serc):
        with pytest.raises(ValueError, match="unknown options"):
            open_hive(serc).set_xindex("morton", MortonMocIndex, scheme="nested")

    def test_create_variables_matches_fabricate(self, moc_index):
        (variables,) = MortonMocIndex.create_variables(moc_index).values()
        assert variables.dims == ("cells",)
        assert variables.dtype == np.uint64
        np.testing.assert_array_equal(variables.values, moc_index.ranges.fabricate())

    def test_normalize_chunks(self):
        assert _normalize_chunks(None, 96) is None
        assert _normalize_chunks(32, 96) == (32, 32, 32)
        assert _normalize_chunks(40, 96) == (40, 40, 16)
        assert _normalize_chunks((16, 80), 96) == (16, 80)
        with pytest.raises(ValueError, match="do not sum"):
            _normalize_chunks((16, 16), 96)
        with pytest.raises(ValueError, match="positive"):
            _normalize_chunks(0, 96)

    def test_chunked_create_variables(self, envelope):
        pytest.importorskip("dask")
        index = MortonMocIndex.from_moc(envelope, cell_order=CELL_ORDER, dim="cells", chunks=32)
        (variable,) = index.create_variables().values()
        assert variable.chunks == ((32,) * (index.size // 32),)
        np.testing.assert_array_equal(np.asarray(variable.values), index.ranges.fabricate())

    def test_repr(self, moc_index):
        assert "ranges=5" in moc_index._repr_inline_(80)
        assert f"size={moc_index.size}" in moc_index._repr_inline_(80)
        assert f"level={CELL_ORDER}" in repr(moc_index)


class TestSelParity:
    """Same selections -> same positional results as the pandas MortonIndex."""

    def test_scalar_word(self, ds_moc, ds_pandas):
        word = int(ds_pandas["morton"].values[5])
        got, want = ds_moc.sel(morton=word), ds_pandas.sel(morton=word)
        assert int(got["morton"]) == int(want["morton"])
        np.testing.assert_array_equal(got["count"].values, want["count"].values)
        assert "cells" not in got.dims and "cells" not in want.dims

    def test_word_list(self, ds_moc, ds_pandas):
        words = [int(v) for v in ds_pandas["morton"].values[3:9]]
        got, want = ds_moc.sel(morton=words), ds_pandas.sel(morton=words)
        np.testing.assert_array_equal(got["morton"].values, want["morton"].values)
        np.testing.assert_array_equal(got["count"].values, want["count"].values)
        assert isinstance(got.xindexes["morton"], MortonMocIndex)

    def test_word_array(self, ds_moc, ds_pandas):
        words = np.asarray(ds_pandas["morton"].values[10:20], dtype=np.uint64)
        got, want = ds_moc.sel(morton=words), ds_pandas.sel(morton=words)
        np.testing.assert_array_equal(got["morton"].values, want["morton"].values)

    def test_non_monotonic_list_drops_lazy_index(self, ds_moc, ds_pandas):
        words = [int(v) for v in ds_pandas["morton"].values[[7, 2]]]
        with pytest.warns(UserWarning, match=r"not representable.*lazy index was dropped"):
            got = ds_moc.sel(morton=words)
        want = ds_pandas.sel(morton=words)
        np.testing.assert_array_equal(got["morton"].values, want["morton"].values)
        np.testing.assert_array_equal(got["count"].values, want["count"].values)
        assert "morton" not in got.xindexes  # unrepresentable: index dropped

    def test_decimal_strings(self, ds_moc):
        # Decimal-string labels are a moc-side convenience (the pandas index
        # holds raw uint64); parity is against the word form of the same sel.
        cell = SERC_SHARD + "11"
        by_string = ds_moc.sel(morton=cell)
        by_word = ds_moc.sel(morton=convention.morton_word(cell))
        assert int(by_string["morton"]) == int(by_word["morton"])
        by_list = ds_moc.sel(morton=[cell, SERC_SHARD + "12"])
        assert by_list.sizes["cells"] == 2

    def test_missing_label_raises_like_pandas(self, ds_moc, ds_pandas):
        missing = convention.morton_word("-511133311")
        with pytest.raises(KeyError):
            ds_pandas.sel(morton=missing)
        with pytest.raises(KeyError):
            ds_moc.sel(morton=missing)

    def test_method_and_slice_raise(self, ds_moc):
        word = int(ds_moc["morton"].values[0])
        with pytest.raises(ValueError, match="method"):
            ds_moc.sel(morton=word, method="nearest")
        with pytest.raises(TypeError, match="slice"):
            ds_moc.sel(morton=slice(None))


class TestIselParity:
    @pytest.mark.parametrize(
        "indexer",
        [slice(3, 30), slice(None, None, 5), np.asarray([0, 4, 17, 40])],
    )
    def test_kept_index(self, ds_moc, ds_pandas, indexer):
        got, want = ds_moc.isel(cells=indexer), ds_pandas.isel(cells=indexer)
        np.testing.assert_array_equal(got["morton"].values, want["morton"].values)
        np.testing.assert_array_equal(got["count"].values, want["count"].values)
        assert isinstance(got.xindexes["morton"], MortonMocIndex)

    def test_scalar_collapse(self, ds_moc, ds_pandas):
        got, want = ds_moc.isel(cells=7), ds_pandas.isel(cells=7)
        assert int(got["morton"]) == int(want["morton"])
        assert "morton" not in got.xindexes

    def test_non_monotonic_drops_lazy_index(self, ds_moc, ds_pandas):
        indexer = np.asarray([9, 3, 3])
        with pytest.warns(UserWarning, match=r"not representable.*lazy index was dropped"):
            got = ds_moc.isel(cells=indexer)
        want = ds_pandas.isel(cells=indexer)
        np.testing.assert_array_equal(got["morton"].values, want["morton"].values)
        assert "morton" not in got.xindexes

    def test_representable_selects_do_not_warn(self, ds_moc, ds_pandas):
        # Only the unrepresentable (non-monotonic/duplicated) drop paths warn;
        # representable subsets — and the dimensionless scalar collapse — stay
        # silent (finding: warn on the drop, not on ordinary selection).
        words = [int(v) for v in ds_pandas["morton"].values[3:9]]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds_moc.sel(morton=words)  # monotonic list -> index kept
            ds_moc.sel(morton=int(ds_pandas["morton"].values[5]))  # scalar sel
            ds_moc.isel(cells=slice(3, 30))  # unit-step slice -> kept
            ds_moc.isel(cells=np.asarray([0, 4, 17, 40]))  # increasing -> kept
            ds_moc.isel(cells=7)  # scalar collapse: dimensionless, not a drop


class TestAlignment:
    def test_inner_and_outer_parity(self, ds_moc, ds_pandas):
        for join in ("inner", "outer"):
            a_moc, b_moc = xr.align(
                ds_moc.isel(cells=slice(0, 32)), ds_moc.isel(cells=slice(16, 48)), join=join
            )
            a_pd, b_pd = xr.align(
                ds_pandas.isel(cells=slice(0, 32)),
                ds_pandas.isel(cells=slice(16, 48)),
                join=join,
            )
            np.testing.assert_array_equal(a_moc["morton"].values, a_pd["morton"].values)
            np.testing.assert_array_equal(b_moc["morton"].values, b_pd["morton"].values)
            np.testing.assert_array_equal(a_moc["count"].values, a_pd["count"].values)

    def test_join_left_right(self, moc_index):
        left = MortonMocIndex(moc_index.ranges.subset(slice(0, 32)), dim="cells", name="morton")
        right = MortonMocIndex(moc_index.ranges.subset(slice(16, 48)), dim="cells", name="morton")
        assert left.join(right, how="left").equals(left)
        assert left.join(right, how="right").equals(right)
        with pytest.raises(ValueError, match="unsupported join"):
            left.join(right, how="sideways")

    def test_reindex_like_positions(self, moc_index):
        superset = moc_index
        subset = MortonMocIndex(moc_index.ranges.subset(slice(3, 9)), dim="cells", name="morton")
        (positions,) = superset.reindex_like(subset).values()
        np.testing.assert_array_equal(positions, np.arange(3, 9))
        # The other direction: cells absent from the subset read -1.
        (back,) = subset.reindex_like(superset).values()
        assert (back == -1).sum() == superset.size - subset.size

    def test_equals_dim_and_domain(self, moc_index):
        assert moc_index.equals(MortonMocIndex(moc_index.ranges, dim="cells", name="morton"))
        assert not moc_index.equals(MortonMocIndex(moc_index.ranges, dim="other", name="morton"))
        assert not moc_index.equals(
            MortonMocIndex(moc_index.ranges.subset(slice(0, 4)), dim="cells", name="morton")
        )


class TestMixedPairing:
    def test_index_level_pairing_raises_pointed(self, moc_index, ds_pandas):
        pandas_index = ds_pandas.xindexes["morton"]
        with pytest.raises(TypeError, match="index_kind"):
            moc_index.equals(pandas_index)
        with pytest.raises(TypeError, match="index_kind"):
            moc_index.join(pandas_index)
        with pytest.raises(TypeError, match="index_kind"):
            moc_index.reindex_like(pandas_index)

    def test_dataset_level_alignment_raises(self, ds_moc, ds_pandas):
        # xarray's aligner rejects the conflicting index pair before our
        # pointed TypeError can surface; the error still names both indexes.
        with pytest.raises(Exception, match="MortonMocIndex"):
            xr.align(ds_moc, ds_pandas, join="inner")

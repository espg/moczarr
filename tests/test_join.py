"""Cross-resolution truncation join (phase 6) — ``parent_cells``/``join_coarse``.

Everything goes through ``open_hive`` outputs — never raw leaf paths or
basenames — so these tests stay insulated from the zagg#299 leaf rename
(tracking: issue #11). The §5 predicate is the correctness anchor:
``clip2order(coarse_level, fine_words) == coarse_word`` iff the fine decimal
string starts with the coarse one.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moczarr import convention, open_hive, parent_cells

FIXTURE = Path(__file__).parent / "data" / "serc_hive"
#: The six order-6 shards of the SERC fixture (cells at order 8, 16 per shard).
SERC_SHARDS = ["4331244", "4331421", "4331422", "4331424", "4332133", "4332311"]
CELL_ORDER = 8


@pytest.fixture()
def serc():
    return str(FIXTURE)


@pytest.fixture()
def ds(serc):
    return open_hive(serc)


class TestParentCells:
    def test_decimal_truncation_parity(self, ds):
        # The §5 predicate: truncating the packed word == truncating the
        # decimal string to base + level digits.
        words = np.asarray(ds["morton"].values, dtype=np.uint64)
        for level in (5, 6, 7):
            got = parent_cells(words, level)
            decimals = [convention.morton_decimal(int(w)) for w in words]
            want = np.asarray(
                [
                    convention.morton_word(d[: len(convention.decimal_base(d)) + level])
                    for d in decimals
                ],
                dtype=np.uint64,
            )
            np.testing.assert_array_equal(got, want)

    def test_dataset_input_is_groupby_ready(self, ds):
        da = parent_cells(ds, 6)
        assert isinstance(da, xr.DataArray)
        assert da.name == "parent_o6"
        assert da.dims == ds["morton"].dims
        assert da.dtype == np.uint64
        np.testing.assert_array_equal(da.values, parent_cells(np.asarray(ds["morton"].values), 6))
        # Six shards, sixteen order-8 cells each: the parents ARE the shards.
        means = ds.assign_coords({da.name: da}).groupby(da.name).mean()
        assert means.sizes[da.name] == len(SERC_SHARDS)
        np.testing.assert_array_equal(
            means[da.name].values,
            np.sort(np.asarray([convention.morton_word(s) for s in SERC_SHARDS], np.uint64)),
        )

    def test_array_input_returns_words(self, ds):
        words = np.asarray(ds["morton"].values, dtype=np.uint64)
        got = parent_cells(words, 6)
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.uint64
        assert np.unique(got).size == len(SERC_SHARDS)

    def test_moc_and_pandas_indexed_agree(self, serc, ds):
        moc = open_hive(serc, index_kind="moc")
        np.testing.assert_array_equal(parent_cells(moc, 6).values, parent_cells(ds, 6).values)
        dggs = pytest.importorskip("moczarr.dggs")
        decoded = dggs.decode(open_hive(serc))
        np.testing.assert_array_equal(parent_cells(decoded, 6).values, parent_cells(ds, 6).values)

    def test_equal_level_is_identity(self, ds):
        words = np.asarray(ds["morton"].values, dtype=np.uint64)
        np.testing.assert_array_equal(parent_cells(words, CELL_ORDER), words)

    def test_finer_level_raises(self, ds):
        with pytest.raises(ValueError, match="finer than the cells' order 8"):
            parent_cells(ds, 9)

    def test_no_morton_raises(self):
        with pytest.raises(ValueError, match="no 'morton' coordinate"):
            parent_cells(xr.Dataset({"x": ("cells", [1.0])}), 6)

    def test_out_of_range_level_raises(self, ds):
        with pytest.raises(ValueError, match="0..29"):
            parent_cells(ds, 30)

    def test_zero_cells_raises(self):
        with pytest.raises(ValueError, match="zero cells"):
            parent_cells(np.empty(0, dtype=np.uint64), 6)

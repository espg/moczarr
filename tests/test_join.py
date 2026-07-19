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
from conftest import build_many_leaf_store

from moczarr import convention, join_coarse, open_hive, parent_cells

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


@pytest.fixture()
def coarse_grouped(ds):
    """An order-6 coarse dataset built IN-TEST from the fixture itself.

    Grouping the fine cells by their order-6 parents and renaming the group
    coordinate to ``morton`` yields the simplest correct coarse product: the
    join golden is then self-evident (every fine row must carry its own
    shard's aggregate).
    """
    parents = parent_cells(ds, 6)
    return ds.groupby(parents).mean().rename({parents.name: "morton"})


@pytest.fixture()
def coarse_store(tmp_path):
    """A SECOND generator-shaped store at orders 5/6, partially covering SERC.

    Built in ``tmp_path`` via the zagg-free conftest builder (the committed
    fixture is never touched): shard ``433142`` at order 5 with cells at
    order 6, i.e. parents ``4331421..4331424`` — covering three of the six
    SERC shards (``4331244``, ``4332133``, ``4332311`` stay uncovered, so
    partial-coverage fill/drop paths are real). Its one data variable is
    ``count`` (int64, all ones), which also collides with SERC's ``count``.
    """
    return build_many_leaf_store(tmp_path / "coarse", ["433142"], cell_order=6, shard_order=5)


#: SERC shards covered by ``coarse_store`` (children of shard 433142).
COVERED = {"4331421", "4331422", "4331424"}


def _covered_mask(ds):
    parents = parent_cells(np.asarray(ds["morton"].values), 6)
    covered = {convention.morton_word(s) for s in COVERED}
    return np.asarray([int(p) in covered for p in parents])


class TestJoinCoarse:
    def test_grouped_join_back_golden(self, ds, coarse_grouped):
        # Every fine row carries its own shard's aggregate, for every var.
        out = join_coarse(ds, coarse_grouped, suffix="_o6")
        parents = parent_cells(np.asarray(ds["morton"].values), 6)
        rows = {int(w): i for i, w in enumerate(coarse_grouped["morton"].values)}
        take = np.asarray([rows[int(p)] for p in parents])
        for name in coarse_grouped.data_vars:
            np.testing.assert_array_equal(
                out[f"{name}_o6"].values, coarse_grouped[name].values[take]
            )
        # Full coverage: the left join fills nothing.
        assert not np.isnan(out["count_o6"].values).any()
        assert out.sizes["cells"] == ds.sizes["cells"]

    def test_collision_raises_with_suffix_hint(self, ds, coarse_grouped):
        with pytest.raises(ValueError, match=r"collide.*count.*suffix='_o6'"):
            join_coarse(ds, coarse_grouped)

    def test_suffixed_collision_still_raises(self, ds, coarse_grouped):
        fine = ds.assign(count_x=ds["count"])
        with pytest.raises(ValueError, match="count_x"):
            join_coarse(fine, coarse_grouped, suffix="_x")

    def test_variables_subset(self, ds, coarse_grouped):
        out = join_coarse(ds, coarse_grouped, variables=["h_mean"], suffix="_o6")
        assert "h_mean_o6" in out.data_vars
        assert "count_o6" not in out.data_vars
        with pytest.raises(ValueError, match="not in coarse: nope"):
            join_coarse(ds, coarse_grouped, variables=["nope"], suffix="_o6")

    def test_two_store_left_fill(self, ds, coarse_store):
        out = join_coarse(ds, open_hive(coarse_store), suffix="_c")
        covered = _covered_mask(ds)
        assert covered.sum() == 48  # three of six shards, 16 cells each
        joined = out["count_c"].values
        # int64 coarse counts promote to float64 with NaN fill (the xarray
        # reindex/where default) — documented, not invented.
        assert joined.dtype == np.float64
        np.testing.assert_array_equal(joined[covered], 1.0)
        assert np.isnan(joined[~covered]).all()
        assert out.sizes["cells"] == ds.sizes["cells"]

    def test_two_store_inner_drops(self, ds, coarse_store):
        out = join_coarse(ds, open_hive(coarse_store), how="inner", suffix="_c")
        covered = _covered_mask(ds)
        assert out.sizes["cells"] == covered.sum()
        np.testing.assert_array_equal(out["morton"].values, ds["morton"].values[covered])
        # No fill rows: integer dtype survives an inner join.
        assert out["count_c"].dtype == np.int64
        np.testing.assert_array_equal(out["count_c"].values, 1)

    def test_how_validation(self, ds, coarse_grouped):
        with pytest.raises(ValueError, match="'left' or 'inner'"):
            join_coarse(ds, coarse_grouped, how="outer", suffix="_o6")

    def test_coarse_without_morton_raises(self, ds):
        with pytest.raises(ValueError, match="coarse has no 'morton'"):
            join_coarse(ds, xr.Dataset({"x": ("cells", [1.0])}))

    def test_finer_coarse_raises(self, ds, coarse_store):
        # Roles reversed: a "coarse" finer than fine cannot be a parent.
        with pytest.raises(ValueError, match="finer than the cells' order 6"):
            join_coarse(open_hive(coarse_store), ds)

    @pytest.mark.parametrize("fine_kind", ["plain", "moc", "pandas"])
    @pytest.mark.parametrize("coarse_kind", ["plain", "moc", "pandas"])
    def test_index_kind_pairings(self, serc, coarse_store, fine_kind, coarse_kind):
        # Values-level: every pairing joins identically to the plain/plain
        # reference ("plain" = no index at all; "pandas"/"moc" via decode and
        # the lazy index — the moc-indexed coarse exercises the 6b rank path).
        def opened(root, kind):
            if kind == "pandas":
                dggs = pytest.importorskip("moczarr.dggs")
                return dggs.decode(open_hive(root))
            return open_hive(root, index_kind="moc") if kind == "moc" else open_hive(root)

        fine, coarse = opened(serc, fine_kind), opened(coarse_store, coarse_kind)
        reference = join_coarse(open_hive(serc), open_hive(coarse_store), suffix="_c")
        out = join_coarse(fine, coarse, suffix="_c")
        np.testing.assert_array_equal(out["morton"].values, reference["morton"].values)
        np.testing.assert_array_equal(out["count_c"].values, reference["count_c"].values)

    def test_moc_fine_inner_keeps_lazy_index(self, serc, coarse_store):
        import warnings

        from moczarr.moc_index import MortonMocIndex

        fine = open_hive(serc, index_kind="moc")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = join_coarse(fine, open_hive(coarse_store), how="inner", suffix="_c")
        assert isinstance(out.xindexes["morton"], MortonMocIndex)

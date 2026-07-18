"""``MortonRanges`` against the SERC fixture and the opener (phase 5a).

The money property: the interval substrate's ``fabricate()`` is byte-equal
to the morton coordinate the eager opener materializes — the domain is pure
arithmetic (shard subtrees ∩ AOI), no cell arrays read. Everything else is
rank/take round-trip algebra and the ``aoi_mask`` cross-check that pins the
interval-space AOI semantics to the row-mask semantics.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from moczarr import convention, open_hive, store
from moczarr.coverage import aoi_mask, ranges_words
from moczarr.ranges import MortonRanges, _shift_and_marker

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
def domain(envelope):
    return MortonRanges.from_root_coverage(envelope, CELL_ORDER)


def occupied_cell(serc):
    """The SERC shard's one occupied cell (from its bitmap sidecar)."""
    bits = store.read_coverage_bitmap(serc, convention.leaf_path(SERC_SHARD))
    return convention.morton_decimal(int(bits[0]))


class TestRankSpace:
    def test_probe_words_not_unit_stride_but_ranks_are(self):
        # The substrate's founding probe, pinned: packed words across one
        # subtree are NOT unit-stride; word >> shift is.
        shift, marker = _shift_and_marker(CELL_ORDER)
        depth = 2
        words = np.asarray(
            [
                convention.morton_word(SERC_SHARD + convention.rank_tail(r, depth))
                for r in range(4**depth)
            ],
            dtype=np.uint64,
        )
        assert (np.diff(words) != 1).any()  # word space: NOT unit-stride
        ranks = words >> np.uint64(shift)
        assert (np.diff(ranks) == 1).all()  # rank space: unit-stride
        assert (words == (ranks << np.uint64(shift)) | np.uint64(marker)).all()

    def test_rank_contiguity_across_shard_boundaries(self):
        # Rank-consecutive shards have contiguous subtrees: sibling and
        # parent-crossing boundaries both.
        shift, _ = _shift_and_marker(CELL_ORDER)
        for last, first in [("433142244", "433142311"), ("433142444", "433143111")]:
            a = convention.morton_word(last) >> shift
            b = convention.morton_word(first) >> shift
            assert b - a == 1

    def test_from_shards_merges_adjacent_subtrees(self):
        words = [convention.morton_word(s) for s in ("4331422", "4331423")]
        ranges = MortonRanges.from_shards(words, CELL_ORDER)
        assert len(ranges.intervals) == 1
        assert ranges.size == 32


class TestFabricateParity:
    """fabricate() byte-equal to the eager opener's morton coordinate."""

    def test_whole_store(self, serc, domain):
        eager = np.asarray(open_hive(serc)["morton"].values, dtype=np.uint64)
        assert domain.fabricate().tobytes() == eager.tobytes()

    def test_shard_order_aoi(self, serc, domain):
        word = convention.morton_word(SERC_SHARD)
        eager = np.asarray(open_hive(serc, aoi=[SERC_SHARD])["morton"].values, dtype=np.uint64)
        assert domain.intersect([word]).fabricate().tobytes() == eager.tobytes()

    def test_cell_order_aoi(self, serc, domain):
        cell = occupied_cell(serc)
        eager = np.asarray(open_hive(serc, aoi=[cell])["morton"].values, dtype=np.uint64)
        assert domain.intersect([cell]).fabricate().tobytes() == eager.tobytes()

    def test_walk_fallback(self, serc, tmp_path):
        # Root MOC deleted: the domain builds from the walked stamped-shard
        # list — same arithmetic, one construction seam.
        copy = tmp_path / "serc"
        shutil.copytree(FIXTURE, copy)
        (copy / convention.ROOT_COVERAGE_NAME).unlink()
        root = str(copy)
        shards = [
            convention.morton_word(convention.split_leaf_name(rel.rsplit("/", 1)[-1])[0])
            for rel in store.walk_leaves(root)
            if store.read_commit(root, rel) is not None
        ]
        ranges = MortonRanges.from_shards(shards, CELL_ORDER)
        eager = np.asarray(open_hive(root)["morton"].values, dtype=np.uint64)
        assert ranges.fabricate().tobytes() == eager.tobytes()

    def test_envelope_and_shards_constructions_agree(self, envelope):
        by_envelope = MortonRanges.from_root_coverage(envelope, CELL_ORDER)
        by_shards = MortonRanges.from_shards(ranges_words(envelope), CELL_ORDER)
        assert by_envelope.equals(by_shards)


class TestAoiCrossCheck:
    """Interval-space intersect == aoi_mask over the fabricated words."""

    @pytest.mark.parametrize(
        "aoi",
        [
            [SERC_SHARD],  # shard order (coarser than cells)
            ["43314"],  # much coarser
            ["433142211"],  # exactly cell order -> single cell
            ["4331422112"],  # order 9: finer than cells -> containing cell
            [SERC_SHARD, "433124"],  # mixed members
            ["4331422112", "4331423"],  # mixed orders in one cover
        ],
    )
    def test_intersect_equals_aoi_mask(self, domain, aoi):
        words = np.asarray([convention.morton_word(a) for a in aoi], dtype=np.uint64)
        fabricated = domain.fabricate()
        expected = fabricated[aoi_mask(fabricated, words)]
        np.testing.assert_array_equal(domain.intersect(words).fabricate(), expected)

    def test_disjoint_aoi_is_empty(self, domain):
        empty = domain.intersect([convention.morton_word("-511")])
        assert empty.size == 0
        assert empty.fabricate().size == 0


class TestDomainAlgebra:
    def test_rank_take_round_trip(self, domain):
        positions = np.arange(domain.size, dtype=np.int64)
        words = domain.take(positions)
        np.testing.assert_array_equal(words, domain.fabricate())
        np.testing.assert_array_equal(domain.rank(words), positions)

    def test_take_shuffled_and_negative(self, domain):
        rng = np.random.default_rng(5)
        positions = rng.permutation(domain.size)[:7].astype(np.int64)
        np.testing.assert_array_equal(domain.rank(domain.take(positions)), positions)
        assert domain.take([-1])[0] == domain.fabricate()[-1]

    def test_take_out_of_bounds_raises(self, domain):
        with pytest.raises(IndexError):
            domain.take([domain.size])

    def test_member(self, domain):
        fabricated = domain.fabricate()
        assert domain.member(fabricated).all()
        assert not domain.member([convention.morton_word("-511133311")]).any()
        # A word at the WRONG order never counts, even if its rank aliases.
        assert not domain.member([convention.morton_word(SERC_SHARD)]).any()

    def test_rank_of_non_member_raises(self, domain):
        with pytest.raises(KeyError, match="not in the domain"):
            domain.rank([convention.morton_word("-511133311")])

    def test_rank_missing_fill(self, domain):
        words = [int(domain.fabricate()[3]), convention.morton_word("-511133311")]
        np.testing.assert_array_equal(domain.rank(words, missing=-1), [3, -1])

    def test_subset_slice(self, domain):
        fabricated = domain.fabricate()
        for indexer in [slice(None), slice(3, 30), slice(None, None, 3), slice(30, 3, -1)]:
            if (indexer.step or 1) < 0:
                with pytest.raises(ValueError, match="non-increasing"):
                    domain.subset(indexer)
                continue
            np.testing.assert_array_equal(domain.subset(indexer).fabricate(), fabricated[indexer])

    def test_subset_integer_array(self, domain):
        fabricated = domain.fabricate()
        positions = np.asarray([0, 2, 3, 17, domain.size - 1])
        np.testing.assert_array_equal(domain.subset(positions).fabricate(), fabricated[positions])
        with pytest.raises(ValueError, match="non-increasing"):
            domain.subset(np.asarray([3, 1]))
        with pytest.raises(ValueError, match="non-increasing"):
            domain.subset(np.asarray([3, 3]))

    def test_from_cell_words_round_trip(self, domain):
        rebuilt = MortonRanges.from_cell_words(domain.fabricate(), CELL_ORDER)
        assert rebuilt.equals(domain)
        inferred = MortonRanges.from_cell_words(domain.fabricate())
        assert inferred.equals(domain)

    def test_from_cell_words_rejects_disorder(self, domain):
        words = domain.fabricate()
        with pytest.raises(ValueError, match="ascending"):
            MortonRanges.from_cell_words(words[::-1], CELL_ORDER)
        mixed = np.concatenate([[convention.morton_word(SERC_SHARD)], words])
        with pytest.raises(ValueError, match="mixed orders"):
            MortonRanges.from_cell_words(mixed, CELL_ORDER)

    def test_union_intersection(self, domain):
        left = domain.intersect([convention.morton_word(SERC_SHARD)])
        right = domain.intersect([convention.morton_word("4331423")])
        both = left.union(right)
        assert both.size == left.size + right.size
        assert left.intersection(right).size == 0
        assert both.intersection(left).equals(left)

    def test_mixed_order_algebra_raises(self, domain):
        other = MortonRanges.from_shards([convention.morton_word(SERC_SHARD)], 9)
        with pytest.raises(ValueError, match="pyramid seam"):
            domain.union(other)

    def test_empty_domain(self):
        empty = MortonRanges(np.empty((0, 2), dtype=np.uint64), CELL_ORDER)
        assert empty.size == 0
        assert not empty.member([convention.morton_word(SERC_SHARD + "11")]).any()
        assert empty.fabricate().size == 0


class TestValidation:
    def test_malformed_envelope_range_raises(self):
        bad = {"order": 6, "ranges": [["4331423", "4331422"]]}  # reversed
        with pytest.raises(ValueError, match="malformed coverage range"):
            MortonRanges.from_root_coverage(bad, CELL_ORDER)

    def test_envelope_below_cell_order_raises(self):
        bad = {"order": 9, "ranges": [["4331422112", "4331422112"]]}
        with pytest.raises(ValueError, match="below cell_order"):
            MortonRanges.from_root_coverage(bad, CELL_ORDER)

    def test_shard_below_cell_order_raises(self):
        with pytest.raises(ValueError, match="below cell_order"):
            MortonRanges.from_shards([convention.morton_word("4331422112")], CELL_ORDER)

    def test_reversed_interval_raises(self):
        with pytest.raises(ValueError, match="reversed"):
            MortonRanges(np.asarray([[5, 3]], dtype=np.uint64), CELL_ORDER)

    def test_repr(self, domain):
        assert f"order={CELL_ORDER}" in repr(domain)
        assert f"size={domain.size}" in repr(domain)

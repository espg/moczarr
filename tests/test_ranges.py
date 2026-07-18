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


def _subtree_words(shard, cell_order):
    """Independent oracle: every cell of a shard's subtree, enumerated by
    decimal tail (no interval machinery), for cross-checking ``fabricate()``."""
    depth = cell_order - convention.decimal_order(shard)
    return [convention.morton_word(shard + convention.rank_tail(r, depth)) for r in range(4**depth)]


class TestSouthernAndMultiBaseGoldens:
    """Literal-pinned southern + cross-base goldens (no store, pure arithmetic).

    The SERC fixture is a single northern base (4) at order 8, so every other
    ``MortonRanges`` golden rides one northern base. The substrate's most
    counterintuitive behavior lives where that fixture structurally cannot
    reach: the sign encoding (southern bases pack into K blocks *above*
    northern ones), and the ``+6/-1`` base seam. These literals were derived
    once from mortie and hard-coded — a packing change in the southern or
    cross-base regime fails here rather than passing silently on the northern
    fixture (the phase-4 southern-golden precedent, tests/test_fabricate.py).
    """

    # base -5, two rank-adjacent order-7 shards at cell order 8 (depth 1): the
    # subtrees merge into a single K interval; fabricate() is these 8 words.
    SOUTH_SHARDS = ("-51111111", "-51111112")
    SOUTH_INTERVAL = [720896, 720903]
    SOUTH_WORDS = (
        12682136550675316744,
        12682154142861361160,
        12682171735047405576,
        12682189327233449992,
        12682206919419494408,
        12682224511605538824,
        12682242103791583240,
        12682259695977627656,
    )

    # +6/-1 seam: base 6's max shard and base -1's min shard are K-adjacent
    # (base 6 Kmax 458751, base -1 K0 458752), so their subtrees COALESCE into
    # one interval — the seam is contiguous in K, not a break.
    SEAM_SHARDS = ("6444444", "-1111111")
    SEAM_INTERVAL = [458736, 458767]

    # bases 3, 6, -1, -4 (order-6 shards, none at the max/min of its base):
    # NON-adjacent, so four separate intervals ordered northern-then-southern.
    MULTI_SHARDS = ("3444444", "6111111", "-1111111", "-4444444")
    MULTI_INTERVALS = [[262128, 262143], [393216, 393231], [458752, 458767], [720880, 720895]]

    def test_southern_fabricate_literals(self):
        words = [convention.morton_word(s) for s in self.SOUTH_SHARDS]
        ranges = MortonRanges.from_shards(words, CELL_ORDER)
        assert ranges.intervals.tolist() == [self.SOUTH_INTERVAL]
        np.testing.assert_array_equal(
            ranges.fabricate(), np.asarray(self.SOUTH_WORDS, dtype=np.uint64)
        )

    def test_southern_rank_take_round_trip(self):
        words = [convention.morton_word(s) for s in self.SOUTH_SHARDS]
        ranges = MortonRanges.from_shards(words, CELL_ORDER)
        fabricated = np.asarray(self.SOUTH_WORDS, dtype=np.uint64)
        positions = np.arange(ranges.size, dtype=np.int64)
        np.testing.assert_array_equal(ranges.take(positions), fabricated)
        np.testing.assert_array_equal(ranges.rank(fabricated), positions)
        assert ranges.member(fabricated).all()

    def test_plus6_minus1_seam_coalesces(self):
        words = [convention.morton_word(s) for s in self.SEAM_SHARDS]
        ranges = MortonRanges.from_shards(words, CELL_ORDER)
        assert ranges.intervals.tolist() == [self.SEAM_INTERVAL]  # one interval
        assert ranges.size == 32
        oracle = np.asarray(
            sorted(w for sh in self.SEAM_SHARDS for w in _subtree_words(sh, CELL_ORDER)),
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(ranges.fabricate(), oracle)

    def test_multi_base_orders_and_separates(self):
        words = [convention.morton_word(s) for s in self.MULTI_SHARDS]
        ranges = MortonRanges.from_shards(words, CELL_ORDER)
        assert ranges.intervals.tolist() == self.MULTI_INTERVALS
        oracle = np.asarray(
            sorted(w for sh in self.MULTI_SHARDS for w in _subtree_words(sh, CELL_ORDER)),
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(ranges.fabricate(), oracle)


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

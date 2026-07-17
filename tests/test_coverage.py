"""Coverage envelopes: leaf gate, box, bitmap decode, root ranges.

The bitmap wire format is pinned by GOLDEN RAW BYTES ported from zagg's
writer tests (``tests/test_coverage.py::test_golden_raw_bytes``): rank =
ascending packed-word order via the base-4 digit tail, MSB-first per byte.
moczarr only decodes, so the goldens are compressed here and decoded back —
any bit-convention drift between writer and reader fails these first.
"""

import numpy as np
import pytest
from numcodecs import Zstd

from moczarr import convention, coverage

SHARD = "-5112333"
NORTH = "5112333"


@pytest.fixture(params=[SHARD, NORTH])
def shard(request):
    return request.param


def _words(*decimals):
    return np.asarray([convention.morton_word(d) for d in decimals], dtype=np.uint64)


def _stamp(cov):
    return {"spec": "morton-hive/1", "complete": True, "coverage": cov}


def _leaf_cov(**overrides):
    base = {
        "spec": coverage.COVERAGE_SPEC,
        "box": [SHARD + "1", SHARD + "4", None, None],
        "cell_order": 8,
        "source": "worker",
    }
    base.update(overrides)
    return base


class TestLeafEnvelope:
    def test_parses(self):
        assert coverage.parse_leaf_coverage(_stamp(_leaf_cov())) == _leaf_cov()

    @pytest.mark.parametrize(
        "stamp",
        [
            None,  # debris / absent leaf
            {"spec": "morton-hive/1", "complete": True},  # pre-coverage stamp
            _stamp("not-a-dict"),
            _stamp({"spec": "morton-moc/9", "box": []}),  # future spec
        ],
    )
    def test_unusable_reads_absent(self, stamp):
        assert coverage.parse_leaf_coverage(stamp) is None

    def test_box_words_drops_nulls(self):
        words = coverage.box_words(_leaf_cov())
        np.testing.assert_array_equal(words, _words(SHARD + "1", SHARD + "4"))
        assert words.dtype == np.uint64


class TestBitmapDecode:
    def _payload(self, raw: bytes) -> bytes:
        return bytes(Zstd(level=3).encode(raw))

    def test_golden_depth2(self, shard):
        # tail "11" = rank 0 -> MSB of byte 0; tail "44" = rank 15 -> LSB of byte 1.
        one = coverage.decode_bitmap(self._payload(b"\x80\x00"), shard, 8)
        np.testing.assert_array_equal(one, _words(shard + "11"))
        last = coverage.decode_bitmap(self._payload(b"\x00\x01"), shard, 8)
        np.testing.assert_array_equal(last, _words(shard + "44"))

    def test_golden_depth3_multi(self, shard):
        # tails 111/114/241/444 -> ranks 0, 3, 28, 63 (zagg's frozen vector).
        decoded = coverage.decode_bitmap(
            self._payload(b"\x90\x00\x00\x08\x00\x00\x00\x01"), shard, 9
        )
        expected = np.sort(_words(*(shard + t for t in ("111", "114", "241", "444"))))
        np.testing.assert_array_equal(decoded, expected)

    def test_word_shard_input(self):
        by_str = coverage.decode_bitmap(self._payload(b"\x80\x00"), SHARD, 8)
        by_word = coverage.decode_bitmap(
            self._payload(b"\x80\x00"), convention.morton_word(SHARD), 8
        )
        np.testing.assert_array_equal(by_str, by_word)

    def test_wrong_size_raises(self, shard):
        # Never zero-pad or truncate: a partial cell set is a false negative.
        with pytest.raises(ValueError, match="refusing to zero-pad"):
            coverage.decode_bitmap(self._payload(b"\xff"), shard, 9)
        with pytest.raises(ValueError, match="refusing to zero-pad"):
            coverage.decode_bitmap(self._payload(b"\xff" * 16), shard, 9)

    def test_depth_zero_raises(self, shard):
        with pytest.raises(ValueError, match="not below"):
            coverage.decode_bitmap(self._payload(b"\x00"), shard, 6)


def _root(**overrides):
    base = {
        "spec": coverage.COVERAGE_SPEC,
        "encoding": "ranges",
        "order": 3,
        "source": "dispatcher",
        "generated_at": "2026-07-17T00:00:00+00:00",
        "ranges": [["-5111", "-5113"], ["-5121", "-5121"]],
    }
    base.update(overrides)
    return base


class TestRootEnvelope:
    def test_parses(self):
        assert coverage.parse_root_coverage(_root()) == _root()

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "garbage",
            {"spec": "morton-moc/9", "encoding": "ranges", "ranges": []},
            _root(encoding="bitmap"),  # root is ranges-only
        ],
    )
    def test_unusable_reads_absent(self, payload):
        assert coverage.parse_root_coverage(payload) is None


class TestRanges:
    def test_expansion(self):
        words = coverage.ranges_words(_root())
        expected = np.unique(_words("-5111", "-5112", "-5113", "-5121"))
        np.testing.assert_array_equal(words, expected)

    def test_contains(self):
        env = _root()
        for dec in ("-5111", "-5112", "-5113", "-5121"):
            assert coverage.ranges_contain(env, dec)
            assert coverage.ranges_contain(env, convention.morton_word(dec))
        assert not coverage.ranges_contain(env, "-5122")
        assert not coverage.ranges_contain(env, "5111")  # other hemisphere
        assert not coverage.ranges_contain(env, "-51111")  # wrong order

    @pytest.mark.parametrize(
        "bad_range",
        [
            ["-5113", "-5111"],  # reversed endpoints
            ["-5111", "5113"],  # base-crossing
            ["-5111", "-51131"],  # wrong order
        ],
    )
    def test_malformed_raises(self, bad_range):
        with pytest.raises(ValueError, match="malformed coverage range"):
            coverage.ranges_words(_root(ranges=[bad_range]))

    def test_moc_and_interop(self):
        # The expanded words feed mortie's MOC algebra: an AOI given as a
        # PARENT cell intersects all its covered children (cross-order
        # containment is what makes the root MOC an index, not a list).
        from mortie import moc_and

        words = coverage.ranges_words(_root())
        aoi = _words("-511")  # parent of the -5111..-5113 run, not of -5121
        hit = moc_and(words, aoi)
        np.testing.assert_array_equal(np.sort(hit), np.sort(_words("-5111", "-5112", "-5113")))


class TestAoiMask:
    """Row-level AOI containment — NOT ``isin`` against ``moc_and`` output.

    The trap this pins (found live in the phase-3 opener): ``moc_and``
    returns a COMPACTED cover, so when every child of a shard is present the
    intersection is the parent word and identity tests match nothing —
    silently dropping exactly the dense regions.
    """

    def _cells(self, shard):
        # The full depth-1 subtree of `shard` plus one cousin.
        return np.sort(_words(*(shard + d for d in "1234"), "-5121111"))

    def test_parent_member_keeps_whole_subtree(self):
        cells = self._cells("-511111")
        keep = coverage.aoi_mask(cells, _words("-511111"))
        # All four children kept (the compaction trap: moc_and+isin gives 0 here).
        assert keep.sum() == 4
        assert not keep[np.isin(cells, _words("-5121111"))].any()

    def test_equal_order_member(self):
        cells = self._cells("-511111")
        keep = coverage.aoi_mask(cells, _words("-5111112"))
        np.testing.assert_array_equal(cells[keep], _words("-5111112"))

    def test_finer_member_keeps_containing_cell(self):
        cells = self._cells("-511111")
        keep = coverage.aoi_mask(cells, _words("-511111231"))  # 2 orders finer
        np.testing.assert_array_equal(cells[keep], _words("-5111112"))

    def test_disjoint_and_empty(self):
        cells = self._cells("-511111")
        assert coverage.aoi_mask(cells, _words("4331422")).sum() == 0
        assert coverage.aoi_mask(np.asarray([], dtype=np.uint64), _words("-5")).size == 0

    def test_mixed_order_members_union(self):
        cells = self._cells("-511111")
        keep = coverage.aoi_mask(cells, _words("-5111114", "-5121"))
        np.testing.assert_array_equal(cells[keep], np.sort(_words("-5111114", "-5121111")))

"""Coverage envelopes: leaf gate, box, bitmap decode, root ranges.

The bitmap wire format is pinned by GOLDEN RAW BYTES ported from zagg's
writer tests (``tests/test_coverage.py::test_golden_raw_bytes``): rank =
ascending packed-word order via the base-4 digit tail, MSB-first per byte.
moczarr only decodes, so the goldens are compressed here and decoded back —
any bit-convention drift between writer and reader fails these first.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import FakeMoc, FakeToc
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


class TestTemporalSection:
    """The §10 ``zagg-coverage-toc/1`` key discipline and tier-1 decoder
    (issue #45). Conformance on committed bytes is
    ``test_spec_conformance.py::TestTemporalCoverageSection``; here are the
    gate and the malformed-map postures."""

    def _temporal(self, **overrides):
        base = {
            "spec": coverage.TEMPORAL_SPEC,
            "source": "sweep",
            "generated_at": "2026-07-17T00:00:00+00:00",
            "fields": ["h_tdigest"],
            "shards": {"-5111": "123", "-5113": "456"},
        }
        base.update(overrides)
        return base

    def test_known_spec_is_carried_through(self):
        envelope = coverage.parse_root_coverage(_root(temporal=self._temporal()))
        assert envelope["temporal"] == self._temporal()

    @pytest.mark.parametrize(
        "section",
        [
            {"spec": "zagg-coverage-toc/2", "shards": {}},  # future revision
            {"shards": {}},  # unmarked carrier (no spec string): debris
            "garbage",  # non-mapping value
        ],
    )
    def test_unknown_or_malformed_section_reads_absent(self, section):
        # §10 versioned-key discipline: the section reads as absent — the
        # CARRIER stays usable, so the store/sidecar/query is never refused.
        envelope = coverage.parse_root_coverage(_root(temporal=section))
        assert envelope is not None and "temporal" not in envelope

    @pytest.mark.parametrize(
        "shards",
        [
            None,  # sentinel: the required key is absent entirely
            [],  # list: indexes by position, not by shard id
            0,  # int
            "11213",  # string: iterates per CHARACTER, not per shard
        ],
    )
    def test_section_without_a_usable_shards_map_reads_absent(self, shards):
        # §10.1 makes ``shards`` required, so a spec-marked section without a
        # usable mapping is STRUCTURALLY malformed and takes the section's
        # reads-as-absent posture — never a KeyError/TypeError out of an
        # accelerator, which §10 says must never refuse the store.
        section = self._temporal()
        if shards is None:
            section.pop("shards")
        else:
            section["shards"] = shards
        envelope = coverage.parse_root_coverage(_root(temporal=section))
        assert envelope is not None and "temporal" not in envelope
        # And defensively at the decoder, for an envelope assembled by hand.
        shard_words, toc_words = coverage.temporal_shard_words(_root(temporal=section))
        assert shard_words.size == toc_words.size == 0
        assert shard_words.dtype == toc_words.dtype == np.uint64

    @pytest.mark.parametrize("spec", ["zagg-coverage-toc/2", None])
    def test_decoder_strict_checks_the_revision_itself(self, spec):
        # A caller who does json.loads(root_coverage) and hands the envelope
        # straight to the decoder bypasses parse_root_coverage's gate; §10
        # says an unknown revision reads as ABSENT, so it must never decode
        # a future revision's map under v1 semantics.
        section = self._temporal(shards={"-5111": "123"})
        if spec is None:
            section.pop("spec")
        else:
            section["spec"] = spec
        shard_words, toc_words = coverage.temporal_shard_words(_root(temporal=section))
        assert shard_words.size == toc_words.size == 0
        assert shard_words.dtype == toc_words.dtype == np.uint64

    def test_no_section_decodes_to_empty(self):
        # §10 absence rule: no listing is not a claim of no data.
        shard_words, toc_words = coverage.temporal_shard_words(_root())
        assert shard_words.size == toc_words.size == 0
        assert shard_words.dtype == toc_words.dtype == np.uint64

    def test_map_decodes_row_aligned_in_shard_word_order(self):
        # A NORTHERN shard beside southern ones at the same order: "5111"
        # sorts after "-5111" as a string and before it as a packed word, so
        # the two relations genuinely disagree here and this pins the
        # decoder's ordering key rather than merely that SOME sort happened.
        shards = {"-5113": "456", "-5111": "18446744073709551615", "5111": "789"}
        envelope = _root(temporal=self._temporal(shards=shards))
        shard_words, toc_words = coverage.temporal_shard_words(envelope)
        assert shard_words.dtype == toc_words.dtype == np.uint64
        by_word = sorted(shards, key=lambda s: int(_words(s)[0]))
        assert by_word != sorted(shards)
        expected = [(int(_words(s)[0]), int(shards[s])) for s in by_word]
        assert list(zip(shard_words.tolist(), toc_words.tolist())) == expected

    def test_empty_map_decodes_to_empty(self):
        shard_words, toc_words = coverage.temporal_shard_words(
            _root(temporal=self._temporal(shards={}))
        )
        assert shard_words.size == toc_words.size == 0

    @pytest.mark.parametrize(
        "shards",
        [
            {"-51112": "123"},  # key not at the carrier's order
            {"-5111": 123},  # raw JSON number: float-mangled beyond 2^53
            {"-5111": "-1"},  # not a uint64 decimal
            {"-5111": "18446744073709551616"},  # 2^64: out of range
            {"-5111": "junk"},
            {"-5111": "٢"},  # ARABIC-INDIC TWO: isdigit() is Unicode-wide
            {"-5111": "²"},  # SUPERSCRIPT TWO: isdigit(), and int() chokes
        ],
    )
    def test_malformed_map_raises(self, shards):
        # The ranges_words posture: a corrupt cache must never yield a
        # plausible partial answer — validated whole, before any decode.
        envelope = _root(temporal=self._temporal(shards=shards))
        with pytest.raises(ValueError, match="temporal"):
            coverage.temporal_shard_words(envelope)


class TestAsTocWords:
    """The temporal query normalizer (issue #45, ruling 4's permanent grammar)."""

    WINDOW_ISO = ("2019-05-01", "2019-06-01")

    def _internal(self, iso):
        from mortie import from_datetime64

        return int(from_datetime64(np.datetime64(iso)))

    def test_all_window_spellings_encode_the_same_word(self):
        iso = coverage.as_toc_words(self.WINDOW_ISO)
        dt64 = coverage.as_toc_words(tuple(np.datetime64(v) for v in self.WINDOW_ISO))
        ns = coverage.as_toc_words(tuple(self._internal(v) for v in self.WINDOW_ISO))
        assert iso.dtype == np.uint64 and iso.shape == (1,)
        np.testing.assert_array_equal(iso, dt64)
        np.testing.assert_array_equal(iso, ns)

    def test_raw_words_pass_through_and_ints_cast(self):
        words = coverage.as_toc_words(self.WINDOW_ISO)
        np.testing.assert_array_equal(coverage.as_toc_words(words), words)
        np.testing.assert_array_equal(coverage.as_toc_words([int(words[0])]), words)

    def test_word_list_straddling_2_63_casts_exactly(self):
        # numpy widens such a list to float64 (int64 below 2**63, uint64
        # above, float64 across); the exact retry keeps both words.
        old = int(coverage.as_toc_words(("1990-01-01", "1990-02-01"))[0])
        new = int(coverage.as_toc_words(("2020-01-01", "2020-02-01"))[0])
        assert old < 2**63 <= new
        got = coverage.as_toc_words([old, new])
        assert got.dtype == np.uint64
        np.testing.assert_array_equal(got, np.asarray([old, new], dtype=np.uint64))

    @pytest.mark.parametrize("bad", [[-1, 2**64 - 1], [1.5, 2.0**63], [-1]])
    def test_negative_or_float_word_lists_still_raise(self, bad):
        with pytest.raises(ValueError, match="temporal query"):
            coverage.as_toc_words(bad)

    def test_protocol_object(self):
        words = coverage.as_toc_words(self.WINDOW_ISO)
        np.testing.assert_array_equal(coverage.as_toc_words(FakeToc(words)), words)
        np.testing.assert_array_equal(coverage.as_toc_words(FakeToc([int(words[0])])), words)

    def test_protocol_two_word_tuple_reads_as_words_not_a_window(self):
        # A protocol return is WORDS by contract: a 2-tuple must not fall
        # into the (start, end) arm and come back as one span2toc word.
        pair = (np.uint64(1), np.uint64(5))
        got = coverage.as_toc_words(FakeToc(pair))
        np.testing.assert_array_equal(got, np.asarray(pair, dtype=np.uint64))

    def test_degenerate_window_is_an_outward_rounded_range(self):
        # span2toc(t, t) is a RANGE word ~2.15 s wide, not a timestamp: the
        # tuple form has no exact-instant spelling (that is time2toc's job).
        from mortie import toc2time, toc_is_range

        t = self._internal("2019-05-10")
        word = coverage.as_toc_words((t, t))
        assert word.shape == (1,) and toc_is_range(int(word[0]))
        start, end = (int(np.atleast_1d(v)[0]) for v in toc2time(word))
        assert start <= t <= end

    def test_reversed_window_raises_the_kernel_error(self):
        # Degenerate handling is mortie's, not re-invented here.
        with pytest.raises(ValueError, match="after its end"):
            coverage.as_toc_words(("2019-06-01", "2019-05-01"))

    @pytest.mark.parametrize("bad", [[], np.empty(0, dtype=np.uint64), [1.5], "2019-05-01"])
    def test_empty_or_unreadable_queries_raise(self, bad):
        with pytest.raises(ValueError, match="temporal query"):
            coverage.as_toc_words(bad)


class TestTemporalKeep:
    """The §10 pruning asymmetry as a pure mask (issue #45)."""

    def _envelope(self):

        may = coverage.as_toc_words(("2019-05-01", "2019-06-01"))
        # -5111 overlaps May 2019, -5113 is 2003-only, -5121 stays unlisted.
        return _root(
            temporal={
                "spec": coverage.TEMPORAL_SPEC,
                "source": "sweep",
                "generated_at": "2026-07-17T00:00:00+00:00",
                "fields": ["h_tdigest"],
                "shards": {
                    "-5111": str(int(may[0])),
                    "-5113": str(int(coverage.as_toc_words(("2003-01-01", "2003-02-01"))[0])),
                },
            }
        )

    def test_listed_pruned_unlisted_kept(self):
        shards = _words("-5111", "-5113", "-5121")
        keep = coverage.temporal_keep(
            shards, self._envelope(), coverage.as_toc_words(("2019-05-10", "2019-05-20"))
        )
        np.testing.assert_array_equal(keep, [True, False, True])

    def test_timestamp_query_word_is_its_exact_instant(self):
        # A timestamp word decodes to (t, t); the mask must query [t, t+1),
        # never the grammar's match-nothing empty window.
        from mortie import from_datetime64, time2toc

        instant = np.asarray(
            [time2toc(int(from_datetime64(np.datetime64("2019-05-10"))))], dtype=np.uint64
        )
        keep = coverage.temporal_keep(_words("-5111", "-5113"), self._envelope(), instant)
        np.testing.assert_array_equal(keep, [True, False])

    def test_degenerate_window_keeps_the_shard_containing_the_instant(self):
        # The ~2.15 s range word span2toc gives for (t, t) still prunes
        # correctly: the May shard stays, the 2003-only shard goes.
        from mortie import from_datetime64

        t = int(from_datetime64(np.datetime64("2019-05-10")))
        keep = coverage.temporal_keep(
            _words("-5111", "-5113"), self._envelope(), coverage.as_toc_words((t, t))
        )
        np.testing.assert_array_equal(keep, [True, False])

    def test_no_section_keeps_everything(self):
        keep = coverage.temporal_keep(
            _words("-5111", "-5113"), _root(), coverage.as_toc_words(("2019-05-10", "2019-05-20"))
        )
        np.testing.assert_array_equal(keep, [True, True])


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

    #: Pyramid fixture (the zagg#262 overview shape) at THREE distinct orders:
    #: an order-5 parent, its four order-6 children, an order-6 cousin off a
    #: sibling branch, and an order-3 cousin coarser than everything else —
    #: so ``distinct_orders`` is ``{3, 5, 6}`` and the finer-direction inner
    #: loop runs 3 orders for an order-8 member, not 2.
    PARENT = "-511111"  # order 5
    COUSIN = "-5121111"  # order 6, disjoint from PARENT's subtree
    FAR_COUSIN = "-5211"  # order 3, disjoint from both

    def _pyramid(self):
        parent = self.PARENT
        decimals = [parent, *[parent + d for d in "1234"], self.COUSIN, self.FAR_COUSIN]
        return np.sort(_words(*decimals))

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

    def test_mixed_order_cells_pyramid(self):
        # The issue #8 capability unlock (mortie#116): a pyramid-shaped
        # cells array — a parent, its four children, and cousins at two
        # OTHER orders in ONE array (the zagg#262 overview-store shape) —
        # masks correctly against AOIs at several orders. Previously raised.
        parent, cells = self.PARENT, self._pyramid()
        from mortie import orders_of

        assert sorted(int(o) for o in np.unique(orders_of(cells))) == [3, 5, 6]
        # AOI = the parent itself: the parent (equal) + all four children.
        keep = coverage.aoi_mask(cells, _words(parent))
        assert keep.sum() == 5
        assert not keep[np.isin(cells, _words(self.COUSIN, self.FAR_COUSIN))].any()
        # AOI = one child: that child (equal) + the parent (contains it).
        keep = coverage.aoi_mask(cells, _words(parent + "2"))
        np.testing.assert_array_equal(np.sort(cells[keep]), np.sort(_words(parent, parent + "2")))
        # AOI two orders finer than the children: containing child + parent.
        keep = coverage.aoi_mask(cells, _words(parent + "231"))
        np.testing.assert_array_equal(np.sort(cells[keep]), np.sort(_words(parent, parent + "2")))
        # AOI coarser than every cell but ONE: the -5111 subtree at the same
        # order as FAR_COUSIN — which is a different order-3 word, so it is
        # excluded by the equal-order compare, not by the finer-direction loop.
        keep = coverage.aoi_mask(cells, _words("-5111"))
        assert keep.sum() == 5
        assert not keep[np.isin(cells, _words(self.COUSIN, self.FAR_COUSIN))].any()
        # AOI = the order-3 far cousin itself: only that cell.
        keep = coverage.aoi_mask(cells, _words(self.FAR_COUSIN))
        np.testing.assert_array_equal(cells[keep], _words(self.FAR_COUSIN))
        # Disjoint AOI keeps nothing.
        assert coverage.aoi_mask(cells, _words("4331422")).sum() == 0

    def test_mixed_order_cells_and_members(self):
        # Mixed order on BOTH sides at once: one member equal to a child
        # (keeps it + the containing parent), one finer than the cousin
        # (keeps the cousin). The order-8 member's finer-direction loop
        # covers all three coarser cell orders {3, 5, 6}.
        parent, cells = self.PARENT, self._pyramid()
        keep = coverage.aoi_mask(cells, _words(parent + "3", "-51211112"))
        np.testing.assert_array_equal(
            np.sort(cells[keep]), np.sort(_words(parent, parent + "3", self.COUSIN))
        )
        # A member inside the order-3 far cousin selects it via the third
        # inner-loop order.
        keep = coverage.aoi_mask(cells, _words(self.FAR_COUSIN + "1234"))
        np.testing.assert_array_equal(cells[keep], _words(self.FAR_COUSIN))

    def test_member_finer_than_every_cell_skips_whole_array_compare(self, monkeypatch):
        # Perf fast path: a member finer than EVERY cell cannot bit-equal any
        # clipped cell (pass-throughs keep their own order), so the whole-array
        # coarser-or-equal compare is skipped. Pinned two ways — the mask is
        # identical to the un-short-circuited computation, and no clip2order
        # call touches the cells array.
        import mortie

        cells = self._pyramid()
        member = _words(self.PARENT + "2341")  # order 9, finer than all of {3, 5, 6}
        expected = coverage.aoi_mask(cells, member)  # the short-circuited result

        sizes = []
        real_clip = mortie.clip2order

        def counting_clip(order, words):
            sizes.append(np.asarray(words).size)
            return real_clip(order, words)

        monkeypatch.setattr(mortie, "clip2order", counting_clip)
        keep = coverage.aoi_mask(cells, member)
        np.testing.assert_array_equal(keep, expected)
        # One clip per distinct coarser cell order, each on the 1-element
        # member — never on the (7-element) cells array.
        assert sizes == [1, 1, 1]
        # Ground truth: the member's ancestors that are actually present.
        np.testing.assert_array_equal(
            np.sort(cells[keep]), np.sort(_words(self.PARENT, self.PARENT + "2"))
        )

    def test_mixed_order_cells_with_points(self):
        # Point cells may ride alongside COARSER area cells in one array
        # (previously a mixed-order reject; §4 + issue #8): the point keeps
        # via its order-4 ancestor member, the disjoint area cell does not.
        from test_convention import POINT_NORTH_WORD

        cells = np.concatenate([_words("-511111"), np.asarray([POINT_NORTH_WORD], dtype=np.uint64)])
        keep = coverage.aoi_mask(cells, _words("41234"))
        np.testing.assert_array_equal(keep, [False, True])

    def test_pure_point_cells(self):
        # v1 posture (spec §4): points are order-29 members for containment,
        # via their area twins — and a pure-point array is UNIFORM order 29,
        # so the single-order guard accepts it rather than wrongly rejecting.
        from test_convention import POINT_NORTH_WORD, POINT_SOUTH_WORD

        cells = np.asarray([POINT_NORTH_WORD, POINT_SOUTH_WORD], dtype=np.uint64)
        # North point path starts "41234..."; its order-4 ancestor contains it.
        keep = coverage.aoi_mask(cells, _words("41234"))
        np.testing.assert_array_equal(keep, [True, False])

    def test_point_cell_in_order29_area_member(self):
        # An order-29 AREA member on the same path contains the point (twin
        # equality) — raw word equality would miss it (different suffixes).
        from test_convention import AREA29_NORTH_WORD, POINT_NORTH_WORD, POINT_SOUTH_WORD

        cells = np.asarray([POINT_NORTH_WORD, POINT_SOUTH_WORD], dtype=np.uint64)
        keep = coverage.aoi_mask(cells, np.asarray([AREA29_NORTH_WORD], dtype=np.uint64))
        np.testing.assert_array_equal(keep, [True, False])

    def test_mixed_kind_cells_and_point_members(self):
        # Mixed order-29 area + point cells are well-formed (§4: both are
        # order-29 encodings); a POINT member selects the coarser cell that
        # contains its path, like any finer member.
        from test_convention import AREA29_SOUTH_WORD, POINT_NORTH_WORD, POINT_SOUTH_WORD

        cells = np.asarray([POINT_NORTH_WORD, AREA29_SOUTH_WORD], dtype=np.uint64)
        keep = coverage.aoi_mask(cells, np.asarray([POINT_SOUTH_WORD], dtype=np.uint64))
        np.testing.assert_array_equal(keep, [False, True])
        coarse = _words("41234")  # order-4 cells
        keep = coverage.aoi_mask(coarse, np.asarray([POINT_NORTH_WORD], dtype=np.uint64))
        np.testing.assert_array_equal(keep, [True])


class TestAsMocWords:
    """The one AOI boundary normalizer (issue #45) — internal, every seam runs it."""

    def test_protocol_object(self):
        # Duck-typed __morton_moc__() — never an isinstance of a mortie type.
        words = _words(SHARD, NORTH + "11")
        np.testing.assert_array_equal(coverage.as_moc_words(FakeMoc(words)), words)

    @pytest.mark.parametrize("shape", ["uint64", "int_list", "int64", "scalar", "empty"])
    def test_protocol_return_shapes_normalize_identically(self, shape):
        # The protocol result "runs through the arms below" (docstring): the
        # return shape is deliberately unpinned — no mortie import, no
        # isinstance — so every arm has to survive one.
        words = _words(NORTH, NORTH + "11")  # < 2**63, so the signed arm fits
        returns = {
            "uint64": words,
            "int_list": [int(w) for w in words],
            "int64": np.asarray([int(w) for w in words], dtype=np.int64),
            "scalar": np.array(int(words[0]), dtype=np.int64),
            "empty": [],
        }
        expected = {"scalar": words[:1], "empty": np.empty(0, dtype=np.uint64)}.get(shape, words)
        out = coverage.as_moc_words(FakeMoc(returns[shape]))
        assert out.dtype == np.uint64
        np.testing.assert_array_equal(out, expected)

    def test_uint64_passthrough_is_idempotent(self):
        words = _words(SHARD, SHARD + "11")
        out = coverage.as_moc_words(words)
        assert out.dtype == np.uint64
        np.testing.assert_array_equal(out, words)
        np.testing.assert_array_equal(coverage.as_moc_words(out), out)

    def test_decimal_strings_and_scalars(self):
        np.testing.assert_array_equal(coverage.as_moc_words([SHARD, NORTH]), _words(SHARD, NORTH))
        np.testing.assert_array_equal(coverage.as_moc_words(SHARD), _words(SHARD))
        word = int(_words(SHARD)[0])
        np.testing.assert_array_equal(coverage.as_moc_words([word]), _words(SHARD))
        np.testing.assert_array_equal(coverage.as_moc_words(word), _words(SHARD))

    def test_zero_dim_arrays_match_their_one_element_form(self):
        # A 0-d array must yield its ELEMENT, not the container: handing
        # morton_word the ndarray would str-parse the packed word as a
        # decimal id (wrong cell, or ValueError).
        word = int(_words(NORTH)[0])  # < 2**63, so it also fits the signed arm
        for dtype in (np.uint64, np.int64):
            scalar = np.array(word, dtype=dtype)
            assert scalar.ndim == 0
            np.testing.assert_array_equal(
                coverage.as_moc_words(scalar),
                coverage.as_moc_words(np.array([word], dtype=dtype)),
            )
            np.testing.assert_array_equal(coverage.as_moc_words(scalar), _words(NORTH))
        # A southern word (>= 2**63) only lands in uint64, the fast path.
        south = np.array(int(_words(SHARD)[0]), dtype=np.uint64)
        np.testing.assert_array_equal(coverage.as_moc_words(south), _words(SHARD))
        np.testing.assert_array_equal(coverage.as_moc_words(np.array(SHARD)), _words(SHARD))

    def test_empty(self):
        out = coverage.as_moc_words([])
        assert out.dtype == np.uint64 and out.size == 0

    def test_root_coverage_and_accepts_protocol(self):
        envelope = {"order": 6, "ranges": [[SHARD, SHARD]]}
        direct = coverage.root_coverage_and(envelope, _words(SHARD))
        via_fake = coverage.root_coverage_and(envelope, FakeMoc(_words(SHARD)))
        np.testing.assert_array_equal(via_fake, direct)

    def test_box_and_accepts_protocol(self):
        cov = _leaf_cov()
        direct = coverage.box_and(cov, _words(SHARD + "1"))
        via_fake = coverage.box_and(cov, FakeMoc(_words(SHARD + "1")))
        np.testing.assert_array_equal(via_fake, direct)

    def test_aoi_mask_accepts_protocol_and_strings(self):
        cells = _words(SHARD + "11", SHARD + "44", NORTH + "11")
        direct = coverage.aoi_mask(cells, _words(SHARD))
        np.testing.assert_array_equal(coverage.aoi_mask(cells, FakeMoc(_words(SHARD))), direct)
        np.testing.assert_array_equal(coverage.aoi_mask(cells, [SHARD]), direct)


class TestPublicSurface:
    """What this module puts on the package surface, and what it withholds.

    The temporal seam mirrors the spatial one exactly (issue #45, review
    thread on PR #48): ``TEMPORAL_SPEC``/``temporal_shard_words``/
    ``temporal_keep`` are the twins of ``COVERAGE_SPEC``/``ranges_words``/
    ``aoi_mask`` and are public like them, while the two BOUNDARY
    NORMALIZERS stay internal per espg ruling 3 — importable, but off both
    ``moczarr.__all__`` and the docs surface. The docs page's filters are
    keyed to this same split, so a name added to one must move in the other.
    """

    def test_temporal_trio_mirrors_its_spatial_twins(self):
        import moczarr

        for temporal, spatial in (
            ("TEMPORAL_SPEC", "COVERAGE_SPEC"),
            ("temporal_shard_words", "ranges_words"),
            ("temporal_keep", "aoi_mask"),
        ):
            assert spatial in moczarr.__all__  # the twin's standing status
            assert temporal in moczarr.__all__
            assert getattr(moczarr, temporal) is getattr(coverage, temporal)

    def test_boundary_normalizers_stay_internal(self):
        import moczarr

        filters = (Path(__file__).parents[1] / "docs/api/coverage.md").read_text()
        for name in ("as_moc_words", "as_toc_words"):
            assert hasattr(coverage, name)  # importable...
            assert name not in moczarr.__all__  # ...but not on the surface
            assert f'"!^{name}$"' in filters  # ...nor in the rendered docs


class TestCoverageMoc:
    """The typed spatial cast (issue #45 phase 4, mortie >= 0.9.10).

    Direction as ruled: moczarr parses the ``"ranges"`` grammar, mortie
    types the words — there is no ``Moc.from_envelope`` on mortie's side.
    """

    def test_casts_the_expanded_ranges(self):
        import mortie

        envelope = _root()
        moc = coverage.coverage_moc(envelope)
        assert isinstance(moc, mortie.Moc)
        # Same coverage as the raw expansion, which is the claim that
        # matters; the words themselves may differ (see the compaction test).
        expanded = coverage.ranges_words(envelope)
        assert moc.contains(mortie.Moc(expanded))
        assert mortie.Moc(expanded).contains(moc)

    def test_words_are_the_compacted_cover_not_the_expansion(self):
        # `Moc` normalizes eagerly through compress_moc, so a fully
        # occupied sibling set comes back as its PARENT. Pinned because the
        # docstring warns callers not to expect element-for-element
        # equality with ranges_words (the same trap aoi_mask documents).
        envelope = _root(order=7, ranges=[[SHARD + "1", SHARD + "4"]])
        expanded = coverage.ranges_words(envelope)
        moc = coverage.coverage_moc(envelope)
        assert expanded.size == 4
        assert moc.words.size == 1
        assert int(moc.words[0]) == convention.morton_word(SHARD)

    def test_result_feeds_back_into_the_aoi_seams(self):
        # The payoff of the cast: `Moc` satisfies __morton_moc__(), so it
        # is accepted wherever this package takes an AOI cover (phase 1).
        envelope = _root()
        moc = coverage.coverage_moc(envelope)
        np.testing.assert_array_equal(coverage.as_moc_words(moc), moc.words)
        keep = coverage.aoi_mask(coverage.ranges_words(envelope), moc)
        assert keep.all()

    def test_empty_ranges_cast_to_an_empty_moc(self):
        moc = coverage.coverage_moc(_root(ranges=[]))
        assert moc.words.size == 0


class TestCoverageToc:
    """The typed temporal cast (issue #45 phase 5, mortie >= 0.9.10).

    Same direction as :func:`coverage_moc`: moczarr decodes the §10
    section, mortie types the words. The load-bearing case is ABSENCE —
    see :meth:`test_no_section_is_none_not_an_empty_cover`.
    """

    def _envelope(self, shards=None):
        return _root(
            temporal={
                "spec": coverage.TEMPORAL_SPEC,
                "source": "sweep",
                "generated_at": "2026-07-17T00:00:00+00:00",
                "fields": ["h_tdigest"],
                "shards": (
                    {"-5111": str(int(coverage.as_toc_words(("2019-05-01", "2019-06-01"))[0]))}
                    if shards is None
                    else shards
                ),
            }
        )

    def test_casts_the_tier1_words(self):
        import mortie

        toc = coverage.coverage_toc(self._envelope())
        assert isinstance(toc, mortie.Toc)
        _shards, words = coverage.temporal_shard_words(self._envelope())
        assert toc == mortie.Toc(words)
        assert toc.overlaps(mortie.Toc("2019-05-10", "2019-05-20"))
        assert not toc.overlaps(mortie.Toc("2003-01-01", "2003-02-01"))

    def test_no_section_is_none_not_an_empty_cover(self):
        # §10's absence rule, and the reason the signature is `Toc | None`:
        # "publishes no temporal coverage" is NOT "has no data in any
        # window". An empty Toc would answer .overlaps(q) False for every
        # query — the exact false negative §10 forbids — so absence must
        # surface as None and make the caller decide.
        assert coverage.coverage_toc(_root()) is None
        assert coverage.coverage_toc(_root(temporal={"spec": "zagg-coverage-toc/2"})) is None

    def test_words_are_the_normalized_union_not_the_tier1_words(self):
        # `Toc` normalizes eagerly, and normalization is lossy toward
        # coverage: the inner shard's span sits inside the outer's, so its
        # word is ABSORBED and two words in come back as one. The temporal
        # twin of test_words_are_the_compacted_cover_not_the_expansion —
        # pinned because the docstring warns callers that the result is
        # neither row-aligned with nor as long as the tier-1 map.
        shards = {
            "-5111": str(int(coverage.as_toc_words(("2019-01-01", "2020-01-01"))[0])),
            "-5113": str(int(coverage.as_toc_words(("2019-05-01", "2019-06-01"))[0])),
        }
        envelope = self._envelope(shards=shards)
        _listed, tier1 = coverage.temporal_shard_words(envelope)
        toc = coverage.coverage_toc(envelope)
        assert tier1.size == 2
        assert toc.words.size == 1
        assert int(toc.words[0]) in {int(w) for w in tier1}

    def test_empty_map_is_none_too(self):
        # NOT a different claim from the test above: §10.2 rules that a
        # shard the map does not list is *unknown*, never *empty*, so a map
        # listing NOTHING knows nothing about every shard at once — the same
        # epistemic state as no section, and it must take the same arm. An
        # empty Toc here would answer .overlaps(q) False for every query,
        # the very false negative the None arm exists to prevent.
        assert coverage.coverage_toc(self._envelope(shards={})) is None

    def test_result_feeds_back_into_the_when_seam(self):
        # The payoff: Toc satisfies __toc_words__(), so the cast round-trips
        # into the phase-3 temporal seams without any new adapter.
        envelope = self._envelope()
        toc = coverage.coverage_toc(envelope)
        np.testing.assert_array_equal(coverage.as_toc_words(toc), toc.words)
        keep = coverage.temporal_keep(_words("-5111"), envelope, coverage.as_toc_words(toc))
        assert keep.all()

    def test_union_over_several_listed_shards(self):
        # "When does this store hold data" is the union across listed
        # shards — two disjoint campaigns stay two spans (Toc is gappy).
        shards = {
            "-5111": str(int(coverage.as_toc_words(("2019-05-01", "2019-06-01"))[0])),
            "-5113": str(int(coverage.as_toc_words(("2003-01-01", "2003-02-01"))[0])),
        }
        toc = coverage.coverage_toc(self._envelope(shards=shards))
        import mortie

        assert toc.overlaps(mortie.Toc("2019-05-10", "2019-05-20"))
        assert toc.overlaps(mortie.Toc("2003-01-10", "2003-01-20"))
        # The gap between the campaigns is not bridged.
        assert not toc.overlaps(mortie.Toc("2010-01-01", "2010-02-01"))

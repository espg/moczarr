"""``zagg-composition/1`` decoding: word layout, presence floor, count recovery.

The byte layout is pinned by the GOLDEN WORD from zagg spec §3.1
(``0xFF000000FF0000FF`` — a single signal photon with per-surface
confidences ``[4, -1, 0, 3, 1]`` at ``threshold=2``): moczarr only decodes,
so the writer's quantizer is re-derived inline here from the §3.2 contract
(round-half-even + presence floor) — any drift between writer and reader
conventions fails the round-trip first.
"""

from pathlib import Path

import numpy as np
import pytest

from moczarr import composition

FIXTURE = Path(__file__).parent / "data" / "composition"

#: Spec §3.1 golden word: lanes [255, 0, 0, 255, 0, 0, 0, 255], LSB byte first.
GOLDEN_WORD = 0xFF000000FF0000FF
GOLDEN_LANES = [255, 0, 0, 255, 0, 0, 0, 255]


def _pack(lanes):
    """Inverse of unpack for test construction: eight u8 lanes -> one word."""
    return sum(int(k) << (8 * i) for i, k in enumerate(lanes))


def _quantize(counts, n):
    """The §3.2 writer quantizer: round-half-even u8 fractions + presence floor."""
    counts = np.asarray(counts, dtype=np.float64)
    if n <= 0:
        return np.zeros_like(counts, dtype=np.uint64)
    k = np.rint(255.0 * counts / n)
    k = np.where((counts > 0) & (k == 0), 1.0, k)
    return np.clip(k, 0, 255).astype(np.uint64)


class TestUnpack:
    def test_golden_word_pins_lsb_first_byte_order(self):
        lanes = composition.unpack_composition(np.asarray([GOLDEN_WORD], dtype=np.uint64))
        assert lanes.shape == (1, 8) and lanes.dtype == np.uint8
        assert lanes[0].tolist() == GOLDEN_LANES
        # An MSB-first layout would pack the same lanes as a different word.
        assert _pack(GOLDEN_LANES[::-1]) == 0xFF0000FF000000FF != GOLDEN_WORD

    def test_round_trips_pack(self):
        rng = np.random.default_rng(20)
        lanes = rng.integers(0, 256, size=(64, 8), dtype=np.uint8)
        words = np.asarray([_pack(row) for row in lanes], dtype=np.uint64)
        np.testing.assert_array_equal(composition.unpack_composition(words), lanes)

    def test_scalar_passes_through_atleast_1d(self):
        # Vectorized-only API: a bare python int is accepted, output stays 2-D.
        lanes = composition.unpack_composition(GOLDEN_WORD)
        assert lanes.shape == (1, 8)
        assert lanes[0].tolist() == GOLDEN_LANES

    def test_empty_and_zero_words(self):
        assert composition.unpack_composition(np.asarray([], dtype=np.uint64)).shape == (0, 8)
        assert not composition.unpack_composition(np.zeros(3, dtype=np.uint64)).any()

    def test_negative_signed_words_raise_rather_than_wrap(self):
        # int64 -1 would wrap to every lane at 255 — "all eight flags occurred
        # in every photon", the most confidently wrong answer here. Refused.
        with pytest.raises(ValueError, match="negative"):
            composition.unpack_composition(np.asarray([-1], dtype=np.int64))

    def test_non_integer_words_raise_rather_than_truncate(self):
        # A float words array (an xarray path that promoted for a fill value)
        # truncates under an unchecked cast; §3/§7 fix the word as uint64.
        for bad in (np.asarray([1.5]), np.asarray([True]), np.asarray(["1"])):
            with pytest.raises(ValueError, match="integer dtype"):
                composition.unpack_composition(bad)

    def test_non_negative_signed_and_python_ints_are_accepted(self):
        # Strictness stops at what cannot be reinterpreted: a plain list, a
        # non-negative int64 array, and a python int above 2**63 all decode.
        for words in ([GOLDEN_WORD], np.asarray([1, 2], dtype=np.int64), GOLDEN_WORD):
            assert composition.unpack_composition(words).dtype == np.uint8
        assert composition.unpack_composition([GOLDEN_WORD])[0].tolist() == GOLDEN_LANES


class TestCountRecovery:
    def test_exhaustive_round_trip_below_255(self):
        # The §3.2 exactness claim, verified for EVERY (n, c) with n <= 254:
        # quantize then recover, per-lane error < 1/2 so rounding is exact.
        for n in range(1, 255):
            counts = np.arange(n + 1)
            words = np.asarray(
                [_pack([k] + [0] * 7) for k in _quantize(counts, n)], dtype=np.uint64
            )
            recovered = composition.counts_from_composition(words, n)
            np.testing.assert_array_equal(recovered[:, 0], counts, err_msg=f"n={n}")
            assert not recovered[:, 1:].any()

    def test_bounded_estimate_above_254(self):
        # Adopted lean: above N=254 the bounded estimate is returned, never
        # raised — presence stays exact regardless. The bound is COMPOSITE:
        # the writer's quantization costs <= N/510 and the round() in
        # counts_from_composition adds up to another 1/2, so the claim is
        # |recovered - c| <= N/510 + 1/2 for every lane the writer ROUNDED
        # (k = round(255c/N) >= 1). A FLOORED lane (c > 0 quantizing to k=0,
        # forced to 1) is outside it by design and is tested below.
        #
        # Swept over EVERY count, not spot-checked: the excess over a bare
        # N/510 lives in the small-c regime (255c/N just above 1/2) and is
        # n-dependent — at n=1000, c=2 recovers 4, an error of 2 against
        # N/510 = 1.9608, while n=100_000 has no such count at all.
        exceeds_bare_bound = 0
        for n in (255, 256, 509, 510, 765, 1000, 10_000, 100_000):
            counts = np.arange(n + 1)
            rounded = np.rint(255.0 * counts / n) >= 1  # the floor never fired
            words = np.asarray(
                [_pack([k] + [0] * 7) for k in _quantize(counts, n)], dtype=np.uint64
            )
            err = np.abs(composition.counts_from_composition(words, n)[:, 0] - counts)[rounded]
            assert np.all(err <= n / 510 + 0.5), f"n={n}, worst={err.max()}"
            exceeds_bare_bound += int((err > n / 510).sum())
        # The extra 1/2 is load-bearing, not padding: a bare N/510 fails here.
        assert exceeds_bare_bound > 0

    def test_bound_holds_in_the_small_c_regime_at_large_n(self):
        # Same claim where the n-dependence bites hardest: at n=10^6 the worst
        # rounded lane is c=1961 (255c/N = 0.50006 -> k=1), recovering
        # round(10^6/255) = 3922 — an error of 1961, inside N/510 + 1/2
        # (1961.28) and outside a bare N/510 (1960.78).
        n = 1_000_000
        counts = np.arange(5_000)
        words = np.asarray([_pack([k] + [0] * 7) for k in _quantize(counts, n)], dtype=np.uint64)
        err = np.abs(composition.counts_from_composition(words, n)[:, 0] - counts)
        rounded = np.rint(255.0 * counts / n) >= 1
        assert np.all(err[rounded] <= n / 510 + 0.5)
        assert err[1961] == 1961 > n / 510

    def test_floored_lane_recovers_the_floor_estimate(self):
        # One occurrence in 100k: k floors to 1, so recovery says ~N/255 (392),
        # not 0 — the deliberate cost of exact presence at every N.
        n = 100_000
        word = _pack(_quantize([1] + [0] * 7, n))
        recovered = composition.counts_from_composition(np.asarray([word], dtype=np.uint64), n)
        assert recovered[0, 0] == round(n / 255)
        assert composition.presence(np.asarray([word], dtype=np.uint64))[0, 0]

    def test_n_signal_broadcasts_per_cell(self):
        words = np.asarray([GOLDEN_WORD, GOLDEN_WORD, 0], dtype=np.uint64)
        recovered = composition.counts_from_composition(words, np.asarray([1, 10, 7]))
        np.testing.assert_array_equal(recovered[0], [1, 0, 0, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(recovered[1], [10, 0, 0, 10, 0, 0, 0, 10])
        assert not recovered[2].any()

    def test_mismatched_n_signal_raises(self):
        with pytest.raises(ValueError, match="2 cells"):
            composition.counts_from_composition(np.zeros(3, dtype=np.uint64), np.asarray([1, 2]))

    def test_empty_stratum_never_negative(self):
        words = np.asarray([GOLDEN_WORD], dtype=np.uint64)
        for n in (0, -3):
            assert not composition.counts_from_composition(words, n).any()


class TestPresence:
    def test_floor_exact_at_every_n(self):
        # One occurrence out of arbitrarily many quantizes to k >= 1 (the
        # presence floor), so presence reads True at every N — including
        # where count recovery is only an estimate.
        for n in (1, 2, 254, 255, 10_000, 10_000_000):
            word = _pack(_quantize([1, 0, n], n))
            present = composition.presence(np.asarray([word], dtype=np.uint64))[0]
            assert present.tolist() == [True, False, True] + [False] * 5, f"n={n}"

    def test_zero_word_is_all_absent(self):
        assert not composition.presence(np.zeros(2, dtype=np.uint64)).any()

    def test_agrees_with_unpack(self):
        words = np.asarray([GOLDEN_WORD, 0, _pack([0, 1, 0, 0, 255, 0, 0, 0])], dtype=np.uint64)
        np.testing.assert_array_equal(
            composition.presence(words), composition.unpack_composition(words) > 0
        )


#: The §3.3 attrs block (hand-built, conftest raw-object style).
DEFAULT_LANES = ["land", "ocean", "sea_ice", "land_ice", "inland_water", "low", "med", "high"]


def _attrs(**overrides):
    block = {
        "spec": composition.COMPOSITION_SPEC,
        "lanes": list(DEFAULT_LANES),
        "of": "h_tdigest_signal",
        "threshold": 2,
    }
    block.update(overrides)
    return {"composition": block}


class TestAttrsBinding:
    def test_parses(self):
        parsed = composition.parse_composition_attrs(_attrs())
        assert parsed == {"lanes": tuple(DEFAULT_LANES), "of": "h_tdigest_signal", "threshold": 2}

    def test_future_spec_raises(self):
        # Strict gate: a future revision is adopted deliberately, never
        # half-parsed — /2 could re-mean the very same eight bytes.
        with pytest.raises(ValueError, match="zagg-composition/2"):
            composition.parse_composition_attrs(_attrs(spec="zagg-composition/2"))

    @pytest.mark.parametrize("attrs", [{}, {"composition": None}, {"composition": "v1"}, None])
    def test_missing_block_raises(self, attrs):
        with pytest.raises(ValueError, match="no composition block"):
            composition.parse_composition_attrs(attrs)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"lanes": DEFAULT_LANES[:7]},  # wrong count
            {"lanes": DEFAULT_LANES[:7] + ["land"]},  # duplicate name
            {"lanes": "land,ocean"},  # not a sequence of names
        ],
    )
    def test_malformed_lanes_raise(self, overrides):
        with pytest.raises(ValueError, match="composition.lanes"):
            composition.parse_composition_attrs(_attrs(**overrides))

    def test_missing_of_raises(self):
        with pytest.raises(ValueError, match="composition.of"):
            composition.parse_composition_attrs(_attrs(of=None))

    def test_missing_threshold_raises(self):
        attrs = _attrs()
        del attrs["composition"]["threshold"]
        with pytest.raises(ValueError, match="composition.threshold"):
            composition.parse_composition_attrs(attrs)

    @pytest.mark.parametrize("threshold", [None, [2], "2", 2.7, 2.0, True])
    def test_malformed_threshold_raises_value_error(self, threshold):
        # ValueError as documented, and never coerced: §3.3 requires each
        # stratum digest's recorded signal_threshold to AGREE with this value,
        # so int(2.7) -> 2 would fabricate agreement with a cut the store never
        # declared. bool is rejected too (True would read as threshold=1).
        with pytest.raises(ValueError, match="composition.threshold must be an int"):
            composition.parse_composition_attrs(_attrs(threshold=threshold))

    def test_threshold_passes_through_unconverted(self):
        assert composition.parse_composition_attrs(_attrs(threshold=4))["threshold"] == 4

    def test_canonical_lanes_are_the_spec_table(self):
        # The order the /1 gate enforces IS §3.1's table: five per-surface
        # marginals in signal_conf_ph column order, then low/med/high.
        assert composition.COMPOSITION_LANES == tuple(DEFAULT_LANES)
        assert composition.COMPOSITION_LANE_COUNT == 8

    def test_permuted_lanes_raise_as_non_conforming(self):
        # §3.3 fixes the /1 value at exactly the §3.1 order, so a permuted
        # declaration is a non-conforming store, not a relabeling instruction:
        # binding it would read the golden word's byte 3 as "inland_water" and
        # report "land_ice" as absent. Both entry points must refuse.
        words = np.asarray([GOLDEN_WORD], dtype=np.uint64)
        for attrs in (
            _attrs(lanes=DEFAULT_LANES[::-1]),
            _attrs(lanes=DEFAULT_LANES[1:] + ["land"]),
        ):
            with pytest.raises(ValueError, match="non-conforming"):
                composition.parse_composition_attrs(attrs)
            with pytest.raises(ValueError, match="non-conforming"):
                composition.named_lanes(words, attrs)

    def test_named_lanes_bind_to_the_parsed_declaration(self, monkeypatch):
        # The binding is order-DRIVEN, not positional — what lets a future /2
        # re-mean the same eight bytes without touching this function. With the
        # validated block reporting a reversed order, the golden word re-keys:
        # byte 3 becomes "inland_water", "land_ice" moves to byte 4 (=0). A
        # hardcoded zip against COMPOSITION_LANES fails exactly here.
        permuted = tuple(DEFAULT_LANES[::-1])
        monkeypatch.setattr(
            composition,
            "parse_composition_attrs",
            lambda attrs: {"lanes": permuted, "of": "h_tdigest_signal", "threshold": 2},
        )
        named = composition.named_lanes(np.asarray([GOLDEN_WORD], dtype=np.uint64), _attrs())
        assert set(named) == set(DEFAULT_LANES)
        assert named["inland_water"][0] == 255  # byte 3 under the permutation
        assert named["land_ice"][0] == 0  # 255 under the canonical order

    def test_named_lanes_default_order(self):
        named = composition.named_lanes(np.asarray([GOLDEN_WORD], dtype=np.uint64), _attrs())
        assert [int(named[name][0]) for name in DEFAULT_LANES] == GOLDEN_LANES


class TestStoreReadBinding:
    """Bind a REAL zagg-written composition array to its ``of`` weights.

    The array under ``tests/data/composition/`` is vendored from the zagg
    spec-conformance ``kitchen_sink`` fixture (englacial/zagg#346, branch
    ``claude/340-store-spec``, ``tests/data/spec/.../6/composition``) —
    branch-sourced pending that PR's merge, at which point the copy is
    re-checkable against the merged fixture byte-for-byte. The ``n_signal``
    literals below are that fixture's recorded per-cell signal-digest
    weights (``kitchen_sink.expected.json``).
    """

    #: (cell index in the 16-cell dense subtree, n_signal); other cells empty.
    EXPECTED_N = {0: 24, 2: 1, 5: 0, 15: 180}
    GOLDEN_INDEX = 2  # single-photon cell packing exactly the golden word
    NOISE_INDEX = 5  # noise-only cell: empty signal stratum packs to 0

    @pytest.fixture()
    def fixture_array(self):
        import zarr

        return zarr.open_array(str(FIXTURE), mode="r")

    def test_attrs_bind(self, fixture_array):
        parsed = composition.parse_composition_attrs(fixture_array.attrs)
        assert parsed["lanes"] == tuple(DEFAULT_LANES)
        assert parsed["of"] == "h_tdigest_signal"
        assert parsed["threshold"] == 2

    def test_golden_and_noise_cells(self, fixture_array):
        words = fixture_array[:]
        assert int(words[self.GOLDEN_INDEX]) == GOLDEN_WORD
        assert int(words[self.NOISE_INDEX]) == 0
        empty = np.setdiff1d(np.arange(16), list(self.EXPECTED_N))
        assert not words[empty].any()  # fill value everywhere unoccupied

    def test_counts_bind_to_the_of_weights(self, fixture_array):
        # threshold=2 ⇒ the three level lanes partition the signal stratum
        # exactly (spec §3.1), and every n here is <= 254 ⇒ recovery is exact:
        # low + med + high must reproduce each cell's n_signal on the nose.
        words = fixture_array[:]
        named = composition.named_lanes(words, fixture_array.attrs)
        idx = sorted(self.EXPECTED_N)
        n = np.asarray([self.EXPECTED_N[i] for i in idx])
        counts = composition.counts_from_composition(words[idx], n)
        levels = [DEFAULT_LANES.index(name) for name in ("low", "med", "high")]
        np.testing.assert_array_equal(counts[:, levels].sum(axis=1), n)
        # The single-photon cell's lanes ARE that photon's flags.
        golden = counts[idx.index(self.GOLDEN_INDEX)]
        np.testing.assert_array_equal(golden, np.asarray(GOLDEN_LANES) // 255)
        # Surface lanes are overlapping marginals: each bounded by n, and the
        # attrs-named view agrees with presence on occurrence.
        assert (counts <= n[:, None]).all()
        present = composition.presence(words[idx])
        for j, name in enumerate(DEFAULT_LANES):
            np.testing.assert_array_equal(named[name][idx] > 0, present[:, j])

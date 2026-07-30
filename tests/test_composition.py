"""``zagg-composition/1`` decoding: word layout, presence floor, count recovery.

The byte layout is pinned by the GOLDEN WORD from zagg spec §3.1
(``0xFF000000FF0000FF`` — a single signal photon with per-surface
confidences ``[4, -1, 0, 3, 1]`` at ``threshold=2``): moczarr only decodes,
so the writer's quantizer is re-derived inline here from the §3.2 contract
(round-half-even + presence floor) — any drift between writer and reader
conventions fails the round-trip first.
"""

import numpy as np
import pytest

from moczarr import composition

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
        # Adopted lean: above N=254 the bounded ±N/510 estimate is returned,
        # never raised — presence stays exact regardless. The bound holds for
        # every lane the writer rounded (k = round(255c/N) >= 1); a FLOORED
        # lane (c > 0 rounding to k=0, forced to 1) trades count accuracy for
        # exact presence and recovers ~N/255 instead — tested below.
        n = 100_000
        counts = np.asarray([0, 250, 499, 500, 50_000, 99_999, n])
        words = np.asarray([_pack([k] + [0] * 7) for k in _quantize(counts, n)], dtype=np.uint64)
        recovered = composition.counts_from_composition(words, n)[:, 0]
        assert np.all(np.abs(recovered - counts) <= np.ceil(n / 510))

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

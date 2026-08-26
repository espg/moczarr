"""Tests for the HHDC tensor profile (issue #19).

Parity is pinned two ways against the committed SERC strata fixture
(``tests/data/strata_hive``, written by zagg's production writer —
``tools/generate_strata_fixture.py``):

- **Golden parity** (needs the ``zagg`` extra for the digest algebra): the
  committed ``tests/data/strata_goldens/*.npy`` were computed by *zagg's*
  ``readers.tdigest_tensor.read_tensors`` at generation time, so they pin
  bit-identity against a frozen reference even when the installed zagg
  drifts.
- **Live parity** (additionally needs zagg's post-#339 reader surface): the
  two readers run side by side on the same store and must agree exactly —
  end to end and helper by helper, since the reader logic is a port, not
  just the imported algebra. **zagg 0.40.0 is the first release carrying
  that surface, and the extra's declared floor is ``zagg>=0.40``** — so
  this leg is effectively always-on wherever the extra installs. The gates
  stay surface probes rather than version compares, because the surface,
  not the version string, is what the port depends on.
- **``cell_index`` parity** (needs zagg's ``cell_index``, the reference
  implementation issue #52 ported): the two resolve every populated cell of
  the fixture to the same axis index, and refuse a coarser block id the same
  way. Probe-gated for the same reason as the subtree leg — the function
  post-dates the ``>=0.40`` floor.
- **Subtree parity** (additionally needs zagg's ``subtree=`` read surface,
  englacial/zagg#351 — first released in **zagg 0.42.0**): the same
  side-by-side comparison for span-restricted reads, plus the warn/raise
  edges. This one is genuinely conditional — ``needs_zagg_subtree`` skips
  it on 0.40/0.41, where the goldens and the always-on live leg remain the
  enforcement. Raising the extra's floor to ``>=0.42`` would make it
  always-on too; that is a dependency change for sign-off, so the probe
  stays.

The layout kernel, the occupancy predicate and the whole mask channel, the
``open_hive`` no-choke check, and the missing-extra error hint all run
WITHOUT zagg — the mask is moczarr-owned (decoded through
:mod:`moczarr.coverage`), so its contract is pinned in the core leg; the
hint test runs only in a core (no-zagg) environment, so moczarr-only CI
exercises it.
"""

import importlib.util
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
from numcodecs import Zstd
from zarr.storage import LocalStore

from moczarr.convention import (
    COMMIT_ATTR,
    COVERAGE_SIDECAR,
    area29_to_point,
    decimal_base,
    decimal_rank,
    is_point_word,
    morton_decimal,
    morton_word,
)
from moczarr.hhdc import (
    _block_mask,
    _chunk_word,
    _load_occupancy,
    _tensor_side,
    block_rank,
    cell_index,
    has_exact_occupancy,
    rank_to_rowcol,
    read_tensors,
    rowcol_to_rank,
)
from moczarr.ragged import _morton_words, iter_populated_chunks, open_ragged, read_cell, read_ragged

DATA = Path(__file__).parent / "data"
GOLDENS = DATA / "strata_goldens"
EXPECTED = json.loads((DATA / "strata_hive.expected.json").read_text())
LEAF = DATA / "strata_hive" / EXPECTED["leaf"]
GROUP = EXPECTED["group"]
SIGNAL = f"{GROUP}/h_tdigest_signal"
NOISE = f"{GROUP}/h_tdigest_noise"
BLOCK_ORDER = int(EXPECTED["goldens"]["params"]["block_order"])
#: Cells-axis depth of the fixture leaf (16 cells = one order-4 shard subtree).
LEAF_DEPTH = 2
#: The fixture's three nested orders: shard (the cells axis' root), read
#: chunk (4 cells), and cell.
SHARD_ORDER = int(EXPECTED["shard_order"])
CHUNK_ORDER = int(EXPECTED["chunk_order"])
CELL_ORDER = int(EXPECTED["cell_order"])
#: The zagg commit whose ``readers/tdigest_tensor.py`` the ported reader logic
#: in :mod:`moczarr.hhdc` mirrors: the englacial/zagg#336 fold. It carries no
#: tag; zagg **0.40.0** is the first release carrying ``has_exact_occupancy``
#: and ``rank_to_rowcol``, which is what made the live-parity leg reachable
#: from PyPI at all — and the extra's declared floor is ``>=0.40``, so that
#: leg runs wherever the extra installs. The ``subtree=`` leg needs 0.42.0
#: (englacial/zagg#351) and stays probe-gated above the floor.
ZAGG_PORT_COMMIT = "3890cb5"

HAS_ZAGG = importlib.util.find_spec("zagg") is not None
needs_zagg = pytest.mark.skipif(not HAS_ZAGG, reason="needs the moczarr[zagg] extra")


def _zagg_reader():
    """zagg's post-#339 reader surface, or ``None`` (pre-#339 releases lack it)."""
    try:
        from zagg.readers import tdigest_tensor
    except ImportError:
        return None
    return tdigest_tensor if hasattr(tdigest_tensor, "has_exact_occupancy") else None


needs_zagg_reader = pytest.mark.skipif(
    _zagg_reader() is None, reason="needs zagg's post-#339 tdigest_tensor reader"
)


def _store():
    return LocalStore(LEAF)


def _stripped_leaf(tmp_path):
    """A copy of the fixture leaf WITHOUT its commit stamp (the flat-store
    shape): exact occupancy is gone, the mask degrades to 2-state."""
    root = tmp_path / "stripped"
    shutil.copytree(LEAF, root)
    meta_path = root / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta["attributes"].pop(COMMIT_ATTR)
    meta_path.write_text(json.dumps(meta))
    return LocalStore(root)


class TestLayoutKernel:
    """Mortie spec §8 deinterleave, orientation pinned to zagg/gridlook."""

    def test_depth1_golden_orientation(self):
        # rank 0 south corner; x (col) gathers even bits, y (row) odd bits.
        assert tuple(int(v) for v in rank_to_rowcol(0, 1)) == (0, 0)
        assert tuple(int(v) for v in rank_to_rowcol(1, 1)) == (0, 1)
        assert tuple(int(v) for v in rank_to_rowcol(2, 1)) == (1, 0)
        assert tuple(int(v) for v in rank_to_rowcol(3, 1)) == (1, 1)

    def test_not_row_major(self):
        # depth 2, rank 4 = 0b100: row-major would give (1, 0); the
        # deinterleave gives x=2, y=0 -> (row, col) = (0, 2).
        assert tuple(int(v) for v in rank_to_rowcol(4, 2)) == (0, 2)

    def test_roundtrip_all_ranks(self):
        ranks = np.arange(64)
        rows, cols = rank_to_rowcol(ranks, 3)
        np.testing.assert_array_equal(rowcol_to_rank(rows, cols, 3), ranks)


def _occupied_leaf(tmp_path, ranks):
    """A copy of the fixture leaf whose occupancy bitmap marks ``ranks``.

    The sidecar is the frozen O8 encoding (MSB-first bits over the shard
    subtree, zstd), written by hand like ``conftest``'s hive fixture — the
    only way to reach an observed cell that stores NO digest in either
    stratum, which the generated fixture has none of.
    """
    root = tmp_path / "occupied"
    shutil.copytree(LEAF, root)
    bits = np.zeros(4**LEAF_DEPTH, dtype=np.uint8)
    bits[list(ranks)] = 1
    payload = bytes(Zstd(level=3).encode(np.packbits(bits).tobytes()))
    (root / COVERAGE_SIDECAR).write_bytes(payload)
    meta_path = root / "zarr.json"
    meta = json.loads(meta_path.read_text())
    coverage = meta["attributes"][COMMIT_ATTR]["coverage"]
    coverage["nbytes"], coverage["raw_nbytes"] = len(payload), 4**LEAF_DEPTH // 8
    meta_path.write_text(json.dumps(meta))
    return LocalStore(root)


def _cells_axis(store, field=SIGNAL):
    """``(payload array, the cells axis' morton words)`` of an opened leaf."""
    arr, _element = open_ragged(store, field)
    return arr, np.asarray(_morton_words(store, field, 3)[0 : int(arr.shape[0])])


class TestMaskChannel:
    """The mask channel WITHOUT the digest algebra.

    ``read_tensors`` reaches the mask through ``chunk_z_range``, so every
    mask assertion in the parity classes needs the ``zagg`` extra — but
    occupancy decode and placement need no digest algebra at all, and the
    mask is the piece moczarr owns (:func:`moczarr.coverage.decode_bitmap`,
    never zagg's hive path). The ratified englacial/zagg#321/#336 contract is
    pinned here, in the core leg.
    """

    def test_load_occupancy_decodes_the_leaf_sidecar(self):
        store = _store()
        arr, words = _cells_axis(store)
        kind, occupied = _load_occupancy(store, arr, words, SIGNAL)
        assert kind == "bitmap"
        expected = np.sort(np.array([int(c["morton"]) for c in EXPECTED["cells"]], dtype=np.uint64))
        np.testing.assert_array_equal(occupied, expected)

    def test_stripped_stamp_has_no_occupancy(self, tmp_path):
        store = _stripped_leaf(tmp_path)
        arr, words = _cells_axis(store)
        assert _load_occupancy(store, arr, words, SIGNAL) is None
        assert not _block_mask(words, None, LEAF_DEPTH).any()  # the 2-state degrade

    def test_block_mask_places_observed_cells_by_deinterleave(self):
        """Placement is the Z-order deinterleave, spelled out: the fixture's
        observed cells are ranks 0, 2, 5, 13, 15, whose ``(row, col)`` are
        ``(0,0) (1,0) (0,3) (2,3) (3,3)`` — a row-major reshape would put
        them at ``(0,0) (0,2) (1,1) (3,1) (3,3)`` instead."""
        store = _store()
        arr, words = _cells_axis(store)
        mask = _block_mask(words, _load_occupancy(store, arr, words, SIGNAL), LEAF_DEPTH)
        expected = np.zeros((4, 4), dtype=np.uint8)
        for row, col in [(0, 0), (1, 0), (0, 3), (2, 3), (3, 3)]:
            expected[row, col] = 1
        np.testing.assert_array_equal(mask, expected)
        # Same footprint as the committed zagg-computed golden (whose observed
        # cells are 1 or 2 depending on whether the signal digest is stored).
        golden = np.load(GOLDENS / "signal_block_mask.npy")
        np.testing.assert_array_equal(mask.astype(bool), golden.astype(bool))

    def test_observed_cell_with_no_digest_reads_one_on_either_field(self, tmp_path):
        """``1`` is symmetric — "observed, no stored digest on the field being
        read", not "the noise stratum". The fixture pins the crossed case both
        ways already (the golden noise mask carries ``1`` at the signal-only
        cell), and the mask base is stratum-agnostic by construction, so a
        cell observed with BOTH strata empty reads ``1`` on both fields."""
        occupied = [c["index"] for c in EXPECTED["cells"]] + [1]  # cell 1 stores nothing
        store = _occupied_leaf(tmp_path, occupied)
        for field in (SIGNAL, NOISE):
            arr, words = _cells_axis(store, field)
            base = _block_mask(words, _load_occupancy(store, arr, words, field), LEAF_DEPTH)
            assert base[0, 1] == 1  # rank 1 -> (row, col) = (0, 1)
            assert int(base.sum()) == len(occupied)


class TestOccupancyPredicate:
    def test_fixture_leaf_has_exact_occupancy(self):
        assert has_exact_occupancy(_store()) is True

    def test_stripped_stamp_degrades(self, tmp_path):
        assert has_exact_occupancy(_stripped_leaf(tmp_path)) is False


def _digit_oracle(word, block_order):
    """``(rank, order)`` of one word from :func:`moczarr.morton_decimal`.

    The independent oracle the zagg notebook's hand-rolled decode was
    finally pinned against, and the one that caught its bug: the decimal
    id's digit at level ``L``, minus 1, IS that level's rank, so the
    block-local rank is the base-4 value of the digits below the block —
    which is exactly what :func:`moczarr.convention.decimal_rank` (frozen,
    golden-tested in ``test_convention.py``) computes over a synthesized
    tail. Deliberately string arithmetic: it shares no bit with the kernel
    under test.
    """
    decimal = morton_decimal(int(word)).removesuffix("p")  # POINT -> its area twin's digits
    base = decimal_base(decimal)
    tail = decimal[len(base) :]
    return decimal_rank(base + tail[block_order:]), len(tail)


def _sample_ids(seed=52):
    """A deterministic spread of decimal ids over every order 0..29."""
    rng = random.Random(seed)
    ids = []
    for order in range(30):
        for _ in range(3):
            sign = rng.choice(["", "-"])
            tail = "".join(rng.choice("1234") for _ in range(order))
            ids.append(sign + rng.choice("123456") + tail)
    return ids


def _located_words(field=SIGNAL):
    """The fixture's located companion words for its first populated cell.

    Real words written by zagg's production writer (spec §9): mostly
    order-29 POINTs, with coarser AREA fallbacks mixed in — the array
    shape ``block_rank`` exists for.
    """
    _cell, _digest, locations = next(iter(read_ragged(_store(), field, locations=True)))
    return np.asarray(locations, dtype=np.uint64)


class TestPackedWordGeometry:
    """The spec §1 packed-word layout ``block_rank`` decodes, re-verified.

    ``[4-bit prefix | 54-bit body | 6-bit suffix]``: the body carries one
    2-bit digit per level for levels 1..27 (most significant first, so
    level ``L`` sits at bit ``6 + 2*(27 - L)``); levels 28 and 29 have no
    body room and ride the suffix's parent-first preorder band
    ``28 + t28*5 + (t29 + 1)``.
    """

    @pytest.mark.parametrize("level", [1, 2, 13, 26, 27])
    def test_body_levels_carry_two_bits_at_the_documented_place(self, level):
        for digit in range(4):
            tail = "1" * (level - 1) + str(digit + 1) + "1" * (27 - level)
            word = morton_word("1" + tail)
            assert word & 0x3F == 27  # an order-27 id: the suffix IS the order
            assert (word >> (6 + 2 * (27 - level))) & 0x3 == digit

    def test_orders_28_and_29_ride_the_suffix_preorder_band(self):
        for t28 in range(4):
            stem = "1" + "1" * 27 + str(t28 + 1)
            assert morton_word(stem) & 0x3F == 28 + t28 * 5
            for t29 in range(4):
                word = morton_word(stem + str(t29 + 1))
                assert word & 0x3F == 28 + t28 * 5 + t29 + 1
                # ... and the POINT twin re-bases the same body to 48 + t28*4 + t29.
                assert area29_to_point(word) & 0x3F == 48 + t28 * 4 + t29


class TestBlockRank:
    """``block_rank`` against the ``morton_decimal`` digit oracle."""

    @pytest.mark.parametrize("block_order", [0, 1, 4, 6, 27, 28, 29])
    def test_matches_the_digit_oracle_for_every_order(self, block_order):
        ids = [i for i in _sample_ids() if _digit_oracle(morton_word(i), 0)[1] >= block_order]
        words = np.array([morton_word(i) for i in ids], dtype=np.uint64)
        assert words.size  # non-vacuous at every parametrized block order
        rank, order = block_rank(words, block_order)
        expected = [_digit_oracle(w, block_order) for w in words]
        np.testing.assert_array_equal(rank, [r for r, _o in expected])
        np.testing.assert_array_equal(order, [o for _r, o in expected])
        assert bool(np.all(rank < 4 ** (order - block_order)))

    def test_mixed_orders_in_one_array(self):
        # A leaf mixes order-29 located words with coarser cell words; GEDI
        # cell words are order 18. Per-word order is what lets a caller group
        # by depth and make one rank_to_rowcol call per group.
        ids = [EXPECTED["shard"] + "2" * (order - SHARD_ORDER) for order in (6, 18, 29)]
        words = np.array([morton_word(i) for i in ids], dtype=np.uint64)
        rank, order = block_rank(words, SHARD_ORDER)
        np.testing.assert_array_equal(order, [6, 18, 29])
        np.testing.assert_array_equal(rank, [_digit_oracle(w, SHARD_ORDER)[0] for w in words])
        for depth in np.unique(order - SHARD_ORDER):
            group = order - SHARD_ORDER == depth
            rows, cols = rank_to_rowcol(rank[group], int(depth))
            np.testing.assert_array_equal(rowcol_to_rank(rows, cols, int(depth)), rank[group])

    @pytest.mark.parametrize("block_order", [0, 6, 29])
    def test_the_fill_word_is_refused_at_every_block_order(self, block_order):
        # A fill-padded 'morton' coordinate is the array a located reader is
        # most likely to hand over; 0 is not a word (prefix 0 is unreachable).
        word = morton_word(EXPECTED["shard"] + "2" * (29 - SHARD_ORDER))
        words = np.array([word, 0], dtype=np.uint64)
        with pytest.raises(ValueError, match="0 FILL word"):
            block_rank(words, block_order)

    def test_the_fill_word_is_not_reported_as_rank_zero(self):
        # The sentinel's worst case: at block_order 0 a pass-through would
        # answer (0, 0) — a legitimate-looking order-0, rank-0 word.
        with pytest.raises(ValueError, match="0 FILL word"):
            block_rank(np.array([0], dtype=np.uint64), 0)

    def test_point_words_normalize_to_their_area_twin(self):
        area = morton_word("5" + "1234" * 7 + "3")  # order 29
        point = area29_to_point(area)
        assert is_point_word(point)
        both = block_rank(np.array([area, point], dtype=np.uint64), 6)
        assert int(both[0][0]) == int(both[0][1]) == _digit_oracle(area, 6)[0]
        np.testing.assert_array_equal(both[1], [29, 29])

    def test_located_point_words_decode_in_band(self):
        """The zagg ``demo/07_minimal.ipynb`` bug, pinned.

        Most of the fixture's located companion is real order-29 POINT
        words. Read WITHOUT :func:`moczarr.convention.point_to_area29`
        their level-28 digit comes out of the point band
        (``(suffix - 28) // 5`` is 4..7, never 0..3), and the notebook's
        ``rank_to_rowcol(..., 1)`` at the deepest level raised ``rank must
        lie in [0, 4)``.
        """
        words = _located_words()
        points = words[np.asarray(is_point_word(words))]
        assert points.size
        raw_suffix = (points & np.uint64(0x3F)).astype(np.int64)
        assert bool(np.all((raw_suffix - 28) // 5 > 3))  # the un-normalized read
        rank, order = block_rank(points, 28)
        np.testing.assert_array_equal(order, np.full(points.shape, 29))
        rows, cols = rank_to_rowcol(rank, 1)
        assert bool(np.all(rows < 2)) and bool(np.all(cols < 2))
        np.testing.assert_array_equal(rank, [_digit_oracle(w, 28)[0] for w in points])

    def test_a_real_located_companion_is_mixed_kind_and_mixed_order(self):
        # Not a constructed case: the fixture's own companion carries
        # order-29 points alongside coarser AREA fallbacks (its cell word,
        # and order-22 words), all in one array — which is why block_rank
        # returns the per-word order instead of assuming one.
        words = _located_words()
        rank, order = block_rank(words, CELL_ORDER)
        assert set(int(o) for o in order) == {6, 22, 29}
        np.testing.assert_array_equal(rank, [_digit_oracle(w, CELL_ORDER)[0] for w in words])
        for depth in np.unique(order - CELL_ORDER):
            group = order - CELL_ORDER == depth
            rows, cols = rank_to_rowcol(rank[group], int(depth))
            np.testing.assert_array_equal(rowcol_to_rank(rows, cols, int(depth)), rank[group])

    def test_located_words_rank_inside_the_cell_that_stores_them(self):
        # The composed claim a located reader makes: an observation's
        # shard-local rank starts with its CELL's shard-local rank — at
        # every order the companion mixes.
        cell = int(EXPECTED["cells"][0]["morton"])
        words = _located_words()
        rank, order = block_rank(words, SHARD_ORDER)
        cell_rank, _o = block_rank(np.asarray([cell], dtype=np.uint64), SHARD_ORDER)
        prefix = rank >> (2 * (order - CELL_ORDER)).astype(np.uint64)
        np.testing.assert_array_equal(prefix, np.full(words.shape, int(cell_rank[0])))
        assert int(cell_rank[0]) == EXPECTED["cells"][0]["index"]

    def test_cell_words_reproduce_their_cells_axis_position(self):
        """The tie to the tensor path: on a nested-ordered cells axis a
        cell's rank within the axis' shard subtree IS its axis index, which
        is why ``read_tensors`` never decodes a word."""
        _arr, words = _cells_axis(_store())
        written = words != 0
        rank, order = block_rank(words[written], SHARD_ORDER)
        np.testing.assert_array_equal(rank, np.flatnonzero(written))
        np.testing.assert_array_equal(order, np.full(rank.shape, CELL_ORDER))
        # ... and one order finer, the chunk-local rank the sweep reports.
        chunk_rank, _o = block_rank(words[written], CHUNK_ORDER)
        np.testing.assert_array_equal(chunk_rank, np.flatnonzero(written) % 4)

    def test_block_order_finer_than_a_word_raises(self):
        words = np.array([morton_word("43314"), morton_word("4331411")], dtype=np.uint64)
        with pytest.raises(ValueError, match="finer than 1 of the 2 word"):
            block_rank(words, 6)

    def test_negative_block_order_raises(self):
        with pytest.raises(ValueError, match="negative"):
            block_rank(np.array([morton_word("43314")], dtype=np.uint64), -1)

    def test_shapes_and_dtypes(self):
        rank, order = block_rank(np.uint64(morton_word("4331411")), 4)
        assert rank.shape == () and order.shape == ()  # scalar in -> 0-d out
        assert rank.dtype == np.uint64 and order.dtype == np.int64
        rank, order = block_rank(np.empty(0, dtype=np.uint64), 9)
        assert rank.shape == (0,) and order.shape == (0,)
        rank, order = block_rank(np.zeros((2, 3), dtype=np.uint64) + morton_word("4331411"), 4)
        assert rank.shape == order.shape == (2, 3)


class TestCellIndex:
    """``cell_index``: chunk-local ``(row, col)`` -> the ``read_cell`` key."""

    def test_resolves_every_populated_cell_to_its_axis_index(self):
        store = _store()
        arr, words = _cells_axis(store)
        side, depth = _tensor_side(arr, SIGNAL)
        for start, populated in iter_populated_chunks(arr):
            chunk = _chunk_word(words[start : start + side * side], SIGNAL, start)
            for rank, _raw in populated:
                row, col = rank_to_rowcol(rank, depth)
                index = cell_index(store, SIGNAL, chunk, int(row), int(col))
                assert index == start + rank
                assert len(read_cell(store, SIGNAL, index))  # the digest is really there

    def test_the_bare_rank_would_read_the_wrong_cell(self):
        # The failure the function exists to prevent: rank 3 of chunk
        # 433144 is cell 15, but read_cell(3) is in range and silent.
        store = _store()
        assert cell_index(store, SIGNAL, "433144", 1, 1) == 15
        assert int(rowcol_to_rank(1, 1, 1)) == 3

    def test_accepts_both_id_currencies(self):
        store = _store()
        assert cell_index(store, SIGNAL, "433141", 1, 0) == cell_index(
            store, SIGNAL, morton_word("433141"), 1, 0
        )

    def test_composes_with_block_rank(self):
        # The two new primitives end to end: word -> chunk-local rank ->
        # (row, col) -> the global cells-axis index the word came from.
        store = _store()
        for cell in EXPECTED["cells"]:
            word = np.asarray([int(cell["morton"])], dtype=np.uint64)
            rank, order = block_rank(word, CHUNK_ORDER)
            row, col = rank_to_rowcol(rank, int(order[0]) - CHUNK_ORDER)
            chunk = morton_decimal(int(cell["morton"]))[:-1]  # cell order -> chunk order
            assert cell_index(store, SIGNAL, chunk, int(row[0]), int(col[0])) == cell["index"]

    def test_coarser_block_id_raises(self):
        # The order-4 shard word names the whole axis, not one read chunk.
        with pytest.raises(ValueError, match="no stored read chunk"):
            cell_index(_store(), SIGNAL, EXPECTED["shard"], 0, 0)

    def test_unwritten_chunk_id_raises(self):
        # 433143 is the fixture's empty read chunk: well-formed, nothing stored.
        with pytest.raises(ValueError, match="no stored read chunk"):
            cell_index(_store(), SIGNAL, "433143", 0, 0)

    @pytest.mark.parametrize("rowcol", [(2, 0), (0, 2), (-1, 0), (0, -1)])
    def test_row_col_outside_the_block_raises(self, rowcol):
        with pytest.raises(ValueError, match=r"outside .* \(2, 2\) read-chunk block"):
            cell_index(_store(), SIGNAL, "433141", *rowcol)


class TestCellIndexReadPosture:
    """``cell_index``'s read posture, pinned the way ``read_tensors``' is.

    Neither half of the docstring's claim shows up in the ANSWER, so both
    need their own pin: **no digest bytes** — a ``CountingStore`` GET count,
    the same instrument as :class:`TestGetPosture` — and **only the array's
    STORED spans**, never the whole axis. The strata fixture cannot show the
    second (its cells axis is one shard object, so stored spans and whole
    axis are the same interval); ``test_ragged``'s flat store can, because
    its inner chunks 1-2 have no object while the dense ``morton``
    coordinate still names them — a whole-axis scan answers ``4``/``8`` for
    those two chunk ids instead of refusing them.
    """

    def test_reads_no_digest_bytes(self):
        from test_ragged import CountingStore

        store = CountingStore(LEAF)
        store.gets.clear()
        assert cell_index(store, SIGNAL, "433141", 1, 0) == 2
        assert [g for g in store.gets if f"{SIGNAL}/c/" in g[0]] == []
        assert len([g for g in store.gets if f"{GROUP}/morton/c/" in g[0]]) == 1

    def test_an_unstored_span_is_never_searched(self, tmp_path):
        from test_ragged import SHARD, TAILS, build_store

        root, _expected = build_store(tmp_path, sharded=False)
        store = LocalStore(root)
        chunk_ids = [SHARD + TAILS[i][0] for i in (0, 4, 8, 12)]
        assert cell_index(store, "g/field", chunk_ids[0], 0, 0) == 0
        assert cell_index(store, "g/field", chunk_ids[3], 1, 1) == 15
        for absent in chunk_ids[1:3]:
            with pytest.raises(ValueError, match="no stored read chunk"):
                cell_index(store, "g/field", absent, 0, 0)


@pytest.mark.skipif(HAS_ZAGG, reason="runs only in a core (no-zagg) environment")
class TestMissingExtraHint:
    def test_read_tensors_names_the_extra(self):
        with pytest.raises(ImportError, match=r"moczarr\[zagg\]"):
            list(read_tensors(_store(), SIGNAL))


@needs_zagg
class TestGoldenParity:
    """Bit-identity against the committed zagg-computed goldens."""

    @pytest.mark.parametrize("stratum", ["signal", "noise"])
    def test_per_chunk_blocks_match_goldens(self, stratum):
        blocks = list(read_tensors(_store(), f"{GROUP}/h_tdigest_{stratum}"))
        assert len(blocks) == EXPECTED["goldens"][stratum]["n_blocks"]
        tensors = np.load(GOLDENS / f"{stratum}_tensors.npy")
        masks = np.load(GOLDENS / f"{stratum}_masks.npy")
        windows = np.load(GOLDENS / f"{stratum}_windows.npy")
        morton = np.load(GOLDENS / f"{stratum}_morton.npy")
        for i, (tensor, mask, window, word) in enumerate(blocks):
            np.testing.assert_array_equal(tensor, tensors[i])
            np.testing.assert_array_equal(mask, masks[i])
            assert (float(window[0]), float(window[1])) == tuple(windows[i])
            assert int(word) == int(morton[i])

    def test_block_assembly_matches_golden(self):
        """Multi-chunk block: one tensor, one shared window, assembled from
        whole read chunks at ``block_order`` = the shard order."""
        (block,) = list(read_tensors(_store(), SIGNAL, block_order=BLOCK_ORDER))
        tensor, mask, window, word = block
        np.testing.assert_array_equal(tensor, np.load(GOLDENS / "signal_block_tensor.npy"))
        np.testing.assert_array_equal(mask, np.load(GOLDENS / "signal_block_mask.npy"))
        golden_window = np.load(GOLDENS / "signal_block_window.npy")
        assert (float(window[0]), float(window[1])) == tuple(golden_window)
        assert int(word) == int(np.load(GOLDENS / "signal_block_morton.npy"))

    def test_three_state_mask_on_strata(self):
        """The noise-only cell is observed (occupancy) but stores no signal
        digest — mask ``1``; digest cells are ``2``; unobserved ``0``."""
        (_t, mask, _w, _m) = list(read_tensors(_store(), SIGNAL, block_order=BLOCK_ORDER))[0]
        assert set(np.unique(mask)) == {0, 1, 2}

    def test_two_state_degrade_without_stamp(self, tmp_path):
        """No commit stamp (the flat-store shape): the mask degrades to
        ``{0, 2}`` — ``0`` asserts nothing about observation."""
        store = _stripped_leaf(tmp_path)
        assert has_exact_occupancy(store) is False
        for _t, mask, _w, _m in read_tensors(store, SIGNAL):
            assert set(np.unique(mask)) <= {0, 2}


#: Chunk-local nested ranks at every ``(row, col)`` of a DEPTH-2 block — the
#: Z-order curve, written out. A row-major reshape would read
#: ``((0,1,2,3), (4,5,6,7), (8,9,10,11), (12,13,14,15))``; at depth 1 the two
#: agree on all four ranks, which is why the committed goldens (4-cell chunks)
#: cannot see the difference.
DEPTH2_RANKS = ((0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15))
#: Value floor of the synthetic depth-2 digests (see :func:`_deep_digest_store`).
DEEP_BASE = 100.25


def _deep_digest_store(tmp_path, chunks=(0, 2)):
    """A synthetic sharded digest field whose READ CHUNKS hold 16 cells.

    The committed fixture's chunks hold 4 cells (depth 1), where the Z-order
    deinterleave and a row-major reshape agree on every rank — so its
    per-chunk goldens cannot see the layout kernel (only the depth-2 block
    assembly can). This is the smallest geometry where they differ: 64 cells,
    4 read chunks of 16, built from ``test_ragged``'s spec-text-only byte
    recipes so no committed fixture byte moves.

    Every cell of each populated chunk carries a ONE-centroid digest at
    ``DEEP_BASE + rank``, so at the default ``resolution=0.5`` the one
    nonzero bin of a tensor position is ``2 * rank`` — the rank that landed
    there, independent of the reader's own kernel.
    """
    from test_ragged import SHARD, _shard_object, _uint64_meta, _vlen_meta, _write

    from moczarr.convention import morton_word

    tails = [a + b + c for a in "1234" for b in "1234" for c in "1234"]
    attrs = {"ragged": {"spec": "zagg-ragged/1", "element": {"dtype": "float32", "shape": [-1, 2]}}}
    grid = tmp_path / "deep"
    payload = []
    for chunk in range(4):
        if chunk not in chunks:
            payload.append(None)
            continue
        payload.append(
            [
                np.array([[DEEP_BASE + rank, 1.0]], dtype="<f4").tobytes()
                for rank in range(len(DEPTH2_RANKS) ** 2)
            ]
        )
    _write(grid, "g/h_tdigest/zarr.json", _vlen_meta(64, 16, sharded=True, attrs=attrs))
    _write(grid, "g/h_tdigest/c/0", _shard_object(payload))
    _write(grid, "g/morton/zarr.json", _uint64_meta(64))
    words = np.array([morton_word(SHARD + t) for t in tails], dtype="<u8")
    _write(grid, "g/morton/c/0", words.tobytes())
    return LocalStore(grid)


def _gapped_digest_store(tmp_path, chunks=(0, 4, 5, 6, 7, 12)):
    """A synthetic sharded digest field with a GAP between root and chunk order.

    Every other fixture here has the axis root and the read chunk on adjacent
    orders (the strata leaf: root 4, chunk 5; :func:`_deep_digest_store`: root
    6, chunk 7), so no subtree word can sit strictly between them and the
    composed ``max(subtree_order, axis_root_order)`` block-order floor is
    always one of the two ends. This one leaves room: 64 cells (order 9) in 16
    read chunks of 4 (order 8) below ``test_ragged``'s order-6 ``SHARD``, so an
    order-7 subtree is finer than the root, coarser than the chunk, and spans a
    PROPER 16-cell sub-span assembled from 4 read chunks.

    ``chunks`` selects the populated read chunks; the default fills the whole
    order-7 subtree ``SHARD + "2"`` (chunks 4-7, cells 16-31) plus chunks 0 and
    12 OUTSIDE it, so filtering the unrestricted sweep is not a no-op. Cells
    carry ``_deep_digest_store``'s one-centroid weight-1 digests so the block
    algebra stays exact.
    """
    from test_ragged import SHARD, _shard_object, _uint64_meta, _vlen_meta, _write

    from moczarr.convention import morton_word

    tails = [a + b + c for a in "1234" for b in "1234" for c in "1234"]
    attrs = {"ragged": {"spec": "zagg-ragged/1", "element": {"dtype": "float32", "shape": [-1, 2]}}}
    grid = tmp_path / "gapped"
    payload = [
        [
            np.array([[DEEP_BASE + (chunk * 4 + pos) % 16, 1.0]], dtype="<f4").tobytes()
            for pos in range(4)
        ]
        if chunk in chunks
        else None
        for chunk in range(16)
    ]
    _write(grid, "g/h_tdigest/zarr.json", _vlen_meta(64, 4, sharded=True, attrs=attrs))
    _write(grid, "g/h_tdigest/c/0", _shard_object(payload))
    _write(grid, "g/morton/zarr.json", _uint64_meta(64))
    words = np.array([morton_word(SHARD + t) for t in tails], dtype="<u8")
    _write(grid, "g/morton/c/0", words.tobytes())
    return LocalStore(grid)


@needs_zagg
class TestDepth2Placement:
    """Cell placement at depth 2, where Z-order and row-major diverge."""

    def test_every_rank_lands_at_its_deinterleaved_position(self, tmp_path):
        store = _deep_digest_store(tmp_path)
        blocks = list(read_tensors(store, "g/h_tdigest"))
        assert len(blocks) == 2  # one per populated read chunk
        for tensor, _mask, (offset, gain), _word in blocks:
            assert tensor.shape == (4, 4, 128)
            assert (offset, gain) == (float(int(DEEP_BASE)), 0.5)
            for row in range(4):
                for col in range(4):
                    counts = tensor[row, col]
                    assert int(counts.sum()) == 1  # one centroid of weight 1
                    assert int(np.argmax(counts)) == 2 * DEPTH2_RANKS[row][col]


@needs_zagg
class TestSymmetricMask:
    def test_both_strata_empty_cell_is_one_on_both_fields(self, tmp_path):
        """End to end through ``read_tensors``: an observed cell that stores no
        digest in EITHER stratum yields ``1`` on both fields (the symmetric
        reading of the ``1`` state, which the generated fixture cannot show)."""
        occupied = [c["index"] for c in EXPECTED["cells"]] + [1]
        store = _occupied_leaf(tmp_path, occupied)
        for field in (SIGNAL, NOISE):
            (_t, mask, _w, _m) = next(iter(read_tensors(store, field, block_order=BLOCK_ORDER)))
            assert mask[0, 1] == 1


@needs_zagg
class TestGetPosture:
    """``read_tensors`` holds the module's one-GET-per-stored-object posture
    on the ``morton`` coordinate too, not just the digest payload: the
    coordinate is served from one cached window per stored coordinate object,
    never re-sliced per block (which cost ~2 GETs per block — a shard-index
    suffix plus an inner chunk — against one small ``uint64`` array)."""

    @pytest.mark.parametrize("stratum", ["signal", "noise"])
    @pytest.mark.parametrize("block_order", [None, BLOCK_ORDER])
    def test_one_get_per_stored_object(self, stratum, block_order):
        from test_ragged import CountingStore

        field = f"{GROUP}/h_tdigest_{stratum}"
        store = CountingStore(LEAF)
        blocks = list(read_tensors(store, field, block_order=block_order))
        assert blocks  # more than one block per sweep on the per-chunk leg
        assert len([g for g in store.gets if f"{field}/c/" in g[0]]) == 1
        assert len([g for g in store.gets if f"{GROUP}/morton/c/" in g[0]]) == 1


@needs_zagg
class TestFitPolicies:
    """All three behaviours when the trimmed range exceeds the window."""

    def test_raise_names_the_overflow(self):
        with pytest.raises(ValueError, match="exceeds the fixed window"):
            list(read_tensors(_store(), SIGNAL, n_bins=4, resolution=0.5))

    def test_degrade_resolution_stops_at_the_first_fitting_gain(self):
        """A power-of-two gain is not the claim — the claim is that the loop
        widened *just enough*. ``fit="raise"`` is the oracle: the chosen gain
        must fit, and half of it must not (an overshoot by 2x would pass the
        power-of-two check alone)."""
        blocks = list(
            read_tensors(_store(), SIGNAL, n_bins=4, resolution=0.5, fit="degrade_resolution")
        )
        assert blocks
        for tensor, _mask, (_offset, gain), _word in blocks:
            assert tensor.shape[2] == 4  # n_bins fixed
            assert gain > 0.5 and math.log2(gain / 0.5).is_integer()  # doubled from the request
        widest = max(gain for _t, _m, (_o, gain), _w in blocks)
        assert len(list(read_tensors(_store(), SIGNAL, n_bins=4, resolution=widest))) == len(blocks)
        with pytest.raises(ValueError, match="exceeds the fixed window"):
            list(read_tensors(_store(), SIGNAL, n_bins=4, resolution=widest / 2))

    def test_collapse_bins_shrinks_to_power_of_two(self):
        blocks = list(read_tensors(_store(), SIGNAL, fit="collapse_bins"))
        assert blocks
        for tensor, _mask, (offset, gain), _word in blocks:
            n = tensor.shape[2]
            assert gain == 0.5  # resolution fixed
            assert n <= 128 and n & (n - 1) == 0  # power of two, only ever shrinks

    def test_collapse_bins_cannot_grow(self):
        with pytest.raises(ValueError, match="cannot grow the window"):
            list(read_tensors(_store(), SIGNAL, n_bins=4, resolution=0.5, fit="collapse_bins"))


@needs_zagg_reader
class TestPortedSurface:
    """Drift alarm for the PORTED reader logic (not just the imported algebra).

    ``rasterize_cell``, ``chunk_z_range``, the occupancy/mask helpers and the
    ``read_tensors`` body are ports of zagg's ``readers/tdigest_tensor.py``
    (:data:`ZAGG_PORT_COMMIT`), so the parity legs are the only thing holding
    them. Value parity below can only compare what still EXISTS — a rename or
    retirement upstream would silently reduce this file to self-certification,
    which is what these two checks catch. Gated like :class:`TestLiveParity`:
    the surface probe passes from zagg 0.40.0 onward, which the extra's
    ``>=0.40`` floor already guarantees.
    """

    PORTED = (
        "read_tensors",
        "rasterize_cell",
        "chunk_z_range",
        "has_exact_occupancy",
        "rank_to_rowcol",
        "rowcol_to_rank",
        "_block_mask",
        "_load_occupancy",
        "_tensor_side",
        "_chunk_word",
        "_cells_order",
    )

    def test_zagg_still_exposes_every_ported_entry_point(self):
        reference = _zagg_reader()
        missing = [name for name in self.PORTED if not hasattr(reference, name)]
        assert not missing, f"ported from zagg but gone upstream: {missing}"

    def test_ported_helpers_agree_function_by_function(self):
        """Parity at helper granularity, not only through ``read_tensors``:
        a compensating pair of drifts inside the pipeline would cancel out
        end to end but not here."""
        reference = _zagg_reader()
        store = _store()
        arr, words = _cells_axis(store)
        assert _tensor_side(arr, SIGNAL) == reference._tensor_side(arr, SIGNAL)
        assert _chunk_word(words, SIGNAL, 0) == reference._chunk_word(words, SIGNAL, 0)
        ours = _load_occupancy(store, arr, words, SIGNAL)
        theirs = reference._load_occupancy(store, arr, words, SIGNAL)
        assert ours[0] == theirs[0]
        np.testing.assert_array_equal(ours[1], theirs[1])
        np.testing.assert_array_equal(
            _block_mask(words, ours, LEAF_DEPTH),
            reference._block_mask(words, theirs, LEAF_DEPTH),
        )


@needs_zagg_reader
class TestLiveParity:
    """moczarr and zagg read the same store side by side, bit-identically."""

    CASES = {
        "default": {},
        "block": {"block_order": None},  # placeholder replaced in the test
        "degrade": {"n_bins": 4, "resolution": 0.5, "fit": "degrade_resolution"},
        "collapse": {"fit": "collapse_bins"},
        "float32": {"dtype": "float32"},
    }

    @pytest.mark.parametrize("stratum", ["signal", "noise"])
    @pytest.mark.parametrize("case", list(CASES))
    def test_read_tensors_bit_identical(self, stratum, case):
        reference = _zagg_reader()
        kwargs = dict(self.CASES[case])
        if case == "block":
            kwargs["block_order"] = BLOCK_ORDER
        field = f"{GROUP}/h_tdigest_{stratum}"
        ours = list(read_tensors(_store(), field, **kwargs))
        theirs = list(reference.read_tensors(_store(), field, **kwargs))
        assert len(ours) == len(theirs) > 0
        for (t1, m1, w1, i1), (t2, m2, w2, i2) in zip(ours, theirs):
            assert t1.dtype == t2.dtype
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(m1, m2)
            assert w1 == w2
            assert int(i1) == int(i2)

    def test_occupancy_predicate_agrees(self, tmp_path):
        reference = _zagg_reader()
        assert has_exact_occupancy(_store()) == reference.has_exact_occupancy(_store())
        stripped = _stripped_leaf(tmp_path)
        assert has_exact_occupancy(stripped) == reference.has_exact_occupancy(stripped) is False


class TestOpenHiveCheckItem:
    """Issue #19 check-item: ``open_hive`` on a strata store must not choke
    on the vlen arrays it is not decoding (they surface as lazy object-dtype
    variables; decode stays with :mod:`moczarr.ragged`)."""

    def test_open_hive_serves_strata_store(self):
        import moczarr

        ds = moczarr.open_hive(str(DATA / "strata_hive"))
        assert int(ds["count"].sum()) == sum(c["count"] for c in EXPECTED["cells"])
        for name in ("h_tdigest_signal", "h_tdigest_noise"):
            assert name in ds.data_vars
            assert ds[name].dtype == object  # surfaced, not decoded


class TestSubtreeRefusalsAndWarnings:
    """Issue #29 ``subtree=`` gates that need no digest algebra: the span
    resolves (and warns/raises) before any payload byte is decoded."""

    def test_sub_chunk_subtree_refused_pointing_at_read_cell(self):
        # An order-6 word IS one cell — finer than the 4-cell read chunks.
        with pytest.raises(ValueError, match="read_cell"):
            list(read_tensors(_store(), SIGNAL, subtree=EXPECTED["shard"] + "11"))

    def test_malformed_and_too_deep_raise(self):
        with pytest.raises(ValueError):
            list(read_tensors(_store(), SIGNAL, subtree="abc"))
        with pytest.raises(ValueError, match="deeper than"):
            list(read_tensors(_store(), SIGNAL, subtree=EXPECTED["shard"] + "111"))

    def test_out_of_domain_word_warns_once_then_yields_nothing(self):
        import warnings

        sibling = EXPECTED["shard"][:-1] + "3"  # 43313: well-formed, disjoint
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert list(read_tensors(_store(), SIGNAL, subtree=sibling)) == []
        msgs = [str(w.message) for w in rec if "outside this axis" in str(w.message)]
        assert msgs == [
            f"subtree {sibling} is outside this axis' order-4 root "
            f"{EXPECTED['shard']} — yielding nothing"
        ]

    def test_in_domain_empty_subtree_yields_nothing_silently(self):
        import warnings

        empty = EXPECTED["shard"] + str(EXPECTED["empty_chunk"] + 1)  # chunk 2: no digests
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert list(read_tensors(_store(), SIGNAL, subtree=empty)) == []
        assert [w for w in rec if "outside this axis" in str(w.message)] == []


@needs_zagg
class TestSubtreeReadTensors:
    """Issue #29 ``subtree=`` on :func:`read_tensors`.

    Golden contract (zagg's #351 acceptance): the subtree read equals the
    whole-store sweep filtered to the blocks below ``w`` — bit-wise, because
    blocks are whole read chunks: each block derives its z-window from the
    same populated cell set in both reads, so tensor, mask, ``(offset,
    gain)`` and morton id all match exactly.
    """

    @staticmethod
    def _filtered(out, word, order):
        """Whole-sweep blocks whose morton id descends from ``word``."""
        from mortie import clip2order

        return [
            o for o in out if int(clip2order(order, np.asarray([o[3]], dtype=np.uint64))[0]) == word
        ]

    @staticmethod
    def _assert_same(got, expected):
        assert [o[3] for o in got] == [o[3] for o in expected]
        for (t1, m1, s1, _w1), (t2, m2, s2, _w2) in zip(got, expected):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(m1, m2)
            assert s1 == s2

    @pytest.mark.parametrize("stratum", ["signal", "noise"])
    @pytest.mark.parametrize("child", ["1", "4"])
    def test_equals_filtered_sweep_both_currencies(self, stratum, child):
        from moczarr.convention import morton_word

        field = f"{GROUP}/h_tdigest_{stratum}"
        key = EXPECTED["shard"] + child
        sweep = list(read_tensors(_store(), field))
        expected = self._filtered(sweep, morton_word(key), 5)
        assert len(expected) == 1
        for sub in (key, morton_word(key)):
            self._assert_same(list(read_tensors(_store(), field, subtree=sub)), expected)

    def test_root_subtree_equals_the_unrestricted_sweep(self):
        self._assert_same(
            list(read_tensors(_store(), SIGNAL, subtree=EXPECTED["shard"])),
            list(read_tensors(_store(), SIGNAL)),
        )

    def test_block_assembly_inside_the_subtree(self):
        """``subtree`` + ``block_order``: the assembled block equals the same
        block of the unrestricted block-order sweep (the golden one)."""
        got = list(
            read_tensors(_store(), SIGNAL, subtree=EXPECTED["shard"], block_order=BLOCK_ORDER)
        )
        self._assert_same(got, list(read_tensors(_store(), SIGNAL, block_order=BLOCK_ORDER)))
        assert len(got) == 1
        np.testing.assert_array_equal(got[0][0], np.load(GOLDENS / "signal_block_tensor.npy"))

    def test_block_order_coarser_than_the_subtree_raises(self):
        key = EXPECTED["shard"] + "1"  # order 5 == the chunk order
        with pytest.raises(ValueError, match="must be between 5 and the chunk order 5"):
            list(read_tensors(_store(), SIGNAL, subtree=key, block_order=4))
        assert len(list(read_tensors(_store(), SIGNAL, subtree=key, block_order=5))) == 1

    def test_ancestor_word_block_order_floor_is_the_axis_root(self):
        """A word above the leaf's root clips to the whole axis, so the
        floor is the ROOT's order — the documented composed bound
        ``max(subtree_order, axis_root_order) <= block_order <= chunk_order``."""
        above = EXPECTED["shard"][:-1]  # 4331, order 3
        with pytest.raises(ValueError, match="must be between 4 and the chunk order 5"):
            list(read_tensors(_store(), SIGNAL, subtree=above, block_order=3))
        assert len(list(read_tensors(_store(), SIGNAL, subtree=above, block_order=4))) == 1

    def test_block_assembly_over_a_strict_sub_span(self, tmp_path):
        """The composed floor's real case: ``min_order`` is the SUBTREE's own
        order, strictly between the axis root's and the chunk's, with the block
        assembled from SEVERAL read chunks over a PROPER sub-span of the axis.
        On every adjacent-order fixture the floor collapses onto one end (see
        :func:`_gapped_digest_store`), so nothing else covers this.
        """
        from test_ragged import SHARD

        from moczarr.convention import morton_word

        store = _gapped_digest_store(tmp_path)
        field, sub = "g/h_tdigest", SHARD + "2"  # order 7: root 6 < 7 < chunk 8
        word = morton_word(sub)

        # Below the floor: an order-6 block would reach past the 16-cell span.
        with pytest.raises(ValueError, match="must be between 7 and the chunk order 8"):
            list(read_tensors(store, field, subtree=sub, block_order=6))

        # ON the floor: one 4x4 block assembled from the span's 4 read chunks.
        sweep7 = list(read_tensors(store, field, block_order=7))
        assert len(sweep7) == 3  # chunks 0 and 12 sit outside — the filter bites
        got = list(read_tensors(store, field, subtree=sub, block_order=7))
        assert [o[0].shape for o in got] == [(4, 4, 128)]
        self._assert_same(got, self._filtered(sweep7, word, 7))

        # At the chunk order: the four per-chunk blocks of the same span.
        per_chunk = list(read_tensors(store, field, subtree=sub, block_order=8))
        assert [o[0].shape for o in per_chunk] == [(2, 2, 128)] * 4
        self._assert_same(
            per_chunk, self._filtered(list(read_tensors(store, field, block_order=8)), word, 7)
        )

    def test_subtree_keeps_the_three_state_mask(self):
        """The occupancy machinery is untouched by the restriction: chunk 4's
        noise-only cell 13 still reads as ``1`` on the signal field."""
        (block,) = list(read_tensors(_store(), SIGNAL, subtree=EXPECTED["shard"] + "4"))
        _t, mask, _w, _m = block
        assert int((mask == 1).sum()) >= 1


def _zagg_subtree_reader():
    """zagg's reader IF it carries the ``subtree=`` surface (zagg #351/#357)."""
    import inspect

    reference = _zagg_reader()
    if reference is None or "subtree" not in inspect.signature(reference.read_tensors).parameters:
        return None
    return reference


needs_zagg_subtree = pytest.mark.skipif(
    _zagg_subtree_reader() is None,
    reason="needs zagg's subtree read surface (englacial/zagg#351; first released in 0.42)",
)


@needs_zagg
class TestSubtreeGetPosture:
    """The issue #29 acceptance pin: only covering stored objects fetched.

    The fixture leaf is ONE sharded object (K=4 inner chunks of 4 cells), so
    the §1.5 read plan for a chunk-order subtree is the shard-index suffix
    plus only the covering inner chunks — ranged GETs that never reach a
    disjoint chunk's bytes, generalizing ``read_cell``'s 2-GET recipe.
    """

    def test_sharded_fetches_only_the_covering_span(self):
        from test_ragged import CountingStore

        store = CountingStore(LEAF)
        store.gets.clear()
        out = list(read_tensors(store, SIGNAL, subtree=EXPECTED["shard"] + "1"))
        assert len(out) == 1
        data_gets = [g for g in store.gets if f"{SIGNAL}/c/" in g[0]]
        assert all(rng is not None for _k, rng, _n in data_gets)
        (obj_key, _r0, n0), *chunk_gets = data_gets
        k = EXPECTED["chunks_per_shard"]
        assert n0 == 16 * k + 4  # the shard-index suffix
        assert len(chunk_gets) >= 1
        # Decode the shard index: the digest-bearing chunk 4 (cell 15) starts
        # after chunk 1's payload, and no fetched range may reach it.
        obj = (LEAF / f"{SIGNAL}/c/0").read_bytes()
        idx = np.frombuffer(obj[-n0:-4], dtype="<u8").reshape(-1, 2)
        chunk3_start = int(idx[3][0])
        # Bound it inside the object: the §1.5 absent sentinel (2**64 - 1)
        # would leave the range assertion below vacuously true.
        assert 0 < chunk3_start < len(obj)
        assert all(int(r.end) <= chunk3_start for _k, r, _n in chunk_gets)
        # The morton coordinate keeps the one-GET-per-stored-object posture.
        assert len([g for g in store.gets if f"{GROUP}/morton/c/" in g[0]]) == 1


@needs_zagg_subtree
class TestSubtreeLiveParity:
    """moczarr and zagg serve the same ``subtree=`` read bit-identically.

    The issue #29 acceptance criterion, live: both readers on the shared
    committed fixture, same kwargs, byte-equal yields — and the same
    warn/raise behavior at the contract's edges.
    """

    #: The ``above-root`` pair sits ABOVE the leaf's root, so it is the only
    #: case exercising the ``max(subtree_order, axis_root_order)`` clip (the
    #: leaf-store floor) in the parity leg — see
    #: :meth:`TestSubtreeReadTensors.test_ancestor_word_block_order_floor_is_the_axis_root`.
    #: ``BLOCK_ORDER`` is 4, so ``above-root-block`` sits exactly ON that
    #: clipped floor — the value a naive ``subtree_order`` floor would allow
    #: below.
    CASES = ("chunk-1", "chunk-4", "root-word", "root-block", "above-root", "above-root-block")

    @staticmethod
    def _kwargs(case):
        from moczarr.convention import morton_word

        above = EXPECTED["shard"][:-1]  # 4331, order 3 — above the axis root
        kwargs = {
            "chunk-1": {"subtree": EXPECTED["shard"] + "1"},
            "chunk-4": {"subtree": EXPECTED["shard"] + "4"},
            "root-word": {"subtree": morton_word(EXPECTED["shard"])},
            "root-block": {"subtree": EXPECTED["shard"], "block_order": BLOCK_ORDER},
            "above-root": {"subtree": above},
            "above-root-block": {"subtree": above, "block_order": BLOCK_ORDER},
        }
        return kwargs[case]

    @pytest.mark.parametrize("stratum", ["signal", "noise"])
    @pytest.mark.parametrize("case", list(CASES))
    def test_subtree_read_bit_identical(self, stratum, case):
        reference = _zagg_subtree_reader()
        kwargs = self._kwargs(case)
        field = f"{GROUP}/h_tdigest_{stratum}"
        ours = list(read_tensors(_store(), field, **kwargs))
        theirs = list(reference.read_tensors(_store(), field, **kwargs))
        assert len(ours) == len(theirs) > 0
        for (t1, m1, w1, i1), (t2, m2, w2, i2) in zip(ours, theirs):
            assert t1.dtype == t2.dtype
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(m1, m2)
            assert w1 == w2
            assert int(i1) == int(i2)

    def test_disjoint_warning_parity(self):
        import warnings

        reference = _zagg_subtree_reader()
        sibling = EXPECTED["shard"][:-1] + "3"
        messages = []
        for reader in (read_tensors, reference.read_tensors):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                assert list(reader(_store(), SIGNAL, subtree=sibling)) == []
            (w,) = [r for r in rec if "outside this axis" in str(r.message)]
            messages.append(str(w.message))
        assert messages[0] == messages[1]

    def test_refusal_parity(self):
        reference = _zagg_subtree_reader()
        for reader in (read_tensors, reference.read_tensors):
            with pytest.raises(ValueError, match="finer than"):
                list(reader(_store(), SIGNAL, subtree=EXPECTED["shard"] + "11"))
            with pytest.raises(ValueError, match="deeper than"):
                list(reader(_store(), SIGNAL, subtree=EXPECTED["shard"] + "111"))
            # The composed floor is the AXIS ROOT's order, not the (coarser)
            # subtree's: block_order=3 is out of range for the '4331' word.
            with pytest.raises(ValueError, match="must be between 4 and the chunk order 5"):
                list(reader(_store(), SIGNAL, subtree=EXPECTED["shard"][:-1], block_order=3))


def _zagg_cell_index_reader():
    """zagg's reader IF it carries ``cell_index`` (the issue #52 reference)."""
    reference = _zagg_reader()
    return reference if reference is not None and hasattr(reference, "cell_index") else None


@pytest.mark.skipif(
    _zagg_cell_index_reader() is None,
    reason="needs zagg's cell_index, the reference implementation ported in issue #52",
)
class TestCellIndexParity:
    """``cell_index`` side by side with the zagg function it was ported from.

    Probe-gated like the ``subtree=`` leg: ``cell_index`` post-dates the
    extra's declared ``zagg>=0.40`` floor, so the offline
    :class:`TestCellIndex` class stays the enforcement wherever this skips.
    """

    def test_every_populated_cell_agrees(self):
        reference = _zagg_cell_index_reader()
        store = _store()
        arr, words = _cells_axis(store)
        side, depth = _tensor_side(arr, SIGNAL)
        for start, populated in iter_populated_chunks(arr):
            chunk = _chunk_word(words[start : start + side * side], SIGNAL, start)
            for rank, _raw in populated:
                row, col = (int(v) for v in rank_to_rowcol(rank, depth))
                assert cell_index(store, SIGNAL, chunk, row, col) == reference.cell_index(
                    store, SIGNAL, chunk, row, col
                )

    def test_the_coarser_block_id_refusal_agrees(self):
        reference = _zagg_cell_index_reader()
        shard = morton_word(EXPECTED["shard"])
        with pytest.raises(ValueError, match="no stored read chunk"):
            cell_index(_store(), SIGNAL, shard, 0, 0)
        with pytest.raises(ValueError, match="no stored read chunk"):
            reference.cell_index(_store(), SIGNAL, shard, 0, 0)

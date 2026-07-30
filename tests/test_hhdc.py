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
  just the imported algebra. No zagg *release* carries that surface
  (:data:`ZAGG_PORT_COMMIT` is untagged; 0.39.0's reader predates the
  deinterleave), so this leg runs against a zagg **checkout** and skips
  otherwise — the goldens are what enforce the port in a released-zagg
  environment.

The layout kernel, the occupancy predicate and the whole mask channel, the
``open_hive`` no-choke check, and the missing-extra error hint all run
WITHOUT zagg — the mask is moczarr-owned (decoded through
:mod:`moczarr.coverage`), so its contract is pinned in the core leg; the
hint test runs only in a core (no-zagg) environment, so moczarr-only CI
exercises it.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from numcodecs import Zstd
from zarr.storage import LocalStore

from moczarr.convention import COMMIT_ATTR, COVERAGE_SIDECAR
from moczarr.hhdc import (
    _block_mask,
    _chunk_word,
    _load_occupancy,
    _tensor_side,
    has_exact_occupancy,
    rank_to_rowcol,
    read_tensors,
    rowcol_to_rank,
)
from moczarr.ragged import _morton_words, open_ragged

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
#: The zagg commit whose ``readers/tdigest_tensor.py`` the ported reader logic
#: in :mod:`moczarr.hhdc` mirrors: the englacial/zagg#336 fold. It carries no
#: tag, and 0.39.0 (the declared floor) has neither ``has_exact_occupancy`` nor
#: ``rank_to_rowcol`` — hence the checkout-only parity legs below.
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

    def test_degrade_resolution_doubles_gain(self):
        blocks = list(
            read_tensors(_store(), SIGNAL, n_bins=4, resolution=0.5, fit="degrade_resolution")
        )
        assert blocks
        for tensor, _mask, (offset, gain), _word in blocks:
            assert tensor.shape[2] == 4  # n_bins fixed
            # gain is the original resolution doubled some whole number of times
            assert gain >= 0.5 and (gain / 0.5) == 2 ** round(np.log2(gain / 0.5))

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
    which is what these two checks catch. Like :class:`TestLiveParity` they
    are checkout-only: no zagg release carries the post-#339 surface.
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

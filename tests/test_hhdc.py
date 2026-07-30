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
  two readers run side by side on the same store and must agree exactly.

The layout kernel, occupancy predicate, ``open_hive`` no-choke check, and
the missing-extra error hint all run WITHOUT zagg — the hint test runs only
in a core (no-zagg) environment, so moczarr-only CI exercises it.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from zarr.storage import LocalStore

from moczarr.convention import COMMIT_ATTR
from moczarr.hhdc import (
    has_exact_occupancy,
    rank_to_rowcol,
    read_tensors,
    rowcol_to_rank,
)

DATA = Path(__file__).parent / "data"
GOLDENS = DATA / "strata_goldens"
EXPECTED = json.loads((DATA / "strata_hive.expected.json").read_text())
LEAF = DATA / "strata_hive" / EXPECTED["leaf"]
GROUP = EXPECTED["group"]
SIGNAL = f"{GROUP}/h_tdigest_signal"
BLOCK_ORDER = int(EXPECTED["goldens"]["params"]["block_order"])

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

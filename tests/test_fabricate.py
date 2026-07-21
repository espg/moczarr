"""``fabricate_cell_ids``: exact NESTED views from packed morton words.

The money test is golden parity: fabricated ids for the SERC fixture's
``morton`` coordinate byte-for-byte equal ``serc_cell_ids_golden.npy`` — the
``cell_ids`` array the LAST dual-written zagg fixture stored (extracted
before the englacial/zagg#314 writer flip made stores morton-only), so
equality proves the morton-only decision (zagg#262, "NESTED is fabricated,
never stored") loses nothing. The fixture itself is now morton-only; the
dual-written (``emit_cell_ids: true`` transition-hatch) shape is simulated
by ADDING a deliberately deviant stored array to a fixture copy, which pins
that "auto" keeps stored bytes untouched rather than refabricating.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from moczarr import open_hive, store
from moczarr.fabricate import FLOAT64_EXACT_MAX_ORDER, fabricate_cell_ids

FIXTURE = Path(__file__).parent / "data" / "serc_hive"
#: Stored ``cell_ids`` of the final dual-written zagg fixture (pre-#314),
#: frozen as the fabrication parity golden (same shards, same seeds).
GOLDEN_CELL_IDS = Path(__file__).parent / "data" / "serc_cell_ids_golden.npy"
#: First cell of the SERC shard's leaf (decimal "433142211", order 8) and
#: its stored NESTED id — literals pin fabrication against mortie drift too.
GOLDEN_WORD = 5347180132572332040
GOLDEN_NESTED = 238416
#: A production-order (19), southern-hemisphere, negative-polar-base word and
#: its NESTED id — computed once with mortie and hard-coded as a drift guard
#: for the regime the order-8 northern SERC goldens never touch (deep south,
#: lat≈-80°, and the negative-base decimal path at the store's real cell order).
GOLDEN_WORD_O19_SOUTH = 11570383905173274643
GOLDEN_NESTED_O19_SOUTH = 2483716583387
#: Order-29 area/point golden pairs (both hemispheres; words shared with
#: test_convention's §1/§4 goldens) and their NESTED ids: the AREA id at the
#: exact order 29, the POINT id at the order-24 clip (spec §4 — a point has
#: no area claim, so the NESTED view truncates it to the float64-exact
#: ceiling; areas NEVER clip).
AREA29_NORTH_WORD = 4733760060091642285
POINT_NORTH_WORD = 4733760060091642301
NESTED29_NORTH = 895209638871198829
NESTED24_NORTH = 874228162960155
AREA29_SOUTH_WORD = 13712984013617909341
POINT_SOUTH_WORD = 13712984013617909360
NESTED29_SOUTH = 3140015627252765584
NESTED24_SOUTH = 3066421510989028


def golden_cell_ids() -> np.ndarray:
    """The frozen dual-written ``cell_ids`` golden, whole-store concat order."""
    return np.load(GOLDEN_CELL_IDS).astype(np.uint64)


def _dual_written_copy(tmp_path):
    """A fixture copy with a DEVIANT stored ``cell_ids`` array in every leaf.

    Simulates zagg's ``emit_cell_ids: true`` transition-hatch shape (a stored
    NESTED array next to ``morton``) — with fabricated+1 values, so a test
    can tell stored bytes (kept by ``"auto"``) from refabricated ones.
    """
    copy = tmp_path / "serc_dual_written"
    shutil.copytree(FIXTURE, copy)
    group = str(store.read_manifest(str(copy))["cell_order"])
    for rel in store.walk_leaves(str(copy)):
        morton_dir = copy / rel / group / "morton"
        if not (morton_dir / "c" / "0").exists():
            continue  # debris leaf: template metadata only, no chunk written
        words = np.frombuffer((morton_dir / "c" / "0").read_bytes(), dtype="<u8")
        deviant = fabricate_cell_ids(words) + np.uint64(1)
        meta = json.loads((morton_dir / "zarr.json").read_text())
        cell_ids_dir = copy / rel / group / "cell_ids"
        (cell_ids_dir / "c").mkdir(parents=True)
        (cell_ids_dir / "zarr.json").write_text(json.dumps(meta))
        (cell_ids_dir / "c" / "0").write_bytes(deviant.astype("<u8").tobytes())
    return str(copy)


class TestFabricateCellIds:
    def test_known_word(self):
        ids = fabricate_cell_ids([GOLDEN_WORD])
        assert ids.dtype == np.uint64
        assert ids.tolist() == [GOLDEN_NESTED]

    def test_known_word_order19_south(self):
        # Production order (19), southern hemisphere, negative polar base — the
        # regime the order-8 northern SERC goldens never exercise. Literals
        # (not recomputation) so a mortie change at this regime fails a test.
        ids = fabricate_cell_ids([GOLDEN_WORD_O19_SOUTH], level=19)
        assert ids.dtype == np.uint64
        assert ids.tolist() == [GOLDEN_NESTED_O19_SOUTH]

    def test_level_cross_check(self):
        assert fabricate_cell_ids([GOLDEN_WORD], level=8).tolist() == [GOLDEN_NESTED]
        with pytest.raises(ValueError, match="level=6"):
            fabricate_cell_ids([GOLDEN_WORD], level=6)

    def test_empty(self):
        ids = fabricate_cell_ids(np.asarray([], dtype=np.uint64))
        assert ids.size == 0 and ids.dtype == np.uint64

    def test_mixed_orders_raise(self):
        from mortie import clip2order

        words = np.asarray([GOLDEN_WORD], dtype=np.uint64)
        mixed = np.concatenate([words, clip2order(6, words)])
        with pytest.raises(ValueError, match="[Mm]ixed"):
            fabricate_cell_ids(mixed)

    def test_order_above_float64_exact_warns(self):
        from mortie import geo2mort

        word = np.asarray(geo2mort(-80.0, 120.0, order=29), dtype=np.uint64)
        with pytest.warns(UserWarning, match="float64") as record:
            ids = fabricate_cell_ids(word)
        assert ids.dtype == np.uint64
        # Default _stacklevel=3 lands the warning on the direct caller (this
        # test file), not inside moczarr.fabricate.
        assert record[0].filename == __file__
        # The empty case warns off the caller-declared level (no words to
        # derive an order from).
        with pytest.warns(UserWarning, match="float64"):
            fabricate_cell_ids([], level=FLOAT64_EXACT_MAX_ORDER + 1)

    def test_point_words_clip_to_24(self):
        # Point-kind words (spec §1 suffix 48..=63) yield the order-24
        # ancestor's NESTED id — silently (already float64-exact).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ids = fabricate_cell_ids([POINT_NORTH_WORD, POINT_SOUTH_WORD])
        assert ids.dtype == np.uint64
        assert ids.tolist() == [NESTED24_NORTH, NESTED24_SOUTH]

    def test_area29_never_clips_and_warns(self):
        # Genuine order-29 AREA cells keep their exact ids (coarsening an
        # area cell changes the labelled thing — spec §4) behind the >24
        # float64 warning.
        with pytest.warns(UserWarning, match="float64"):
            ids = fabricate_cell_ids([AREA29_NORTH_WORD, AREA29_SOUTH_WORD])
        assert ids.tolist() == [NESTED29_NORTH, NESTED29_SOUTH]

    def test_mixed_area_and_point_words(self):
        # Mixed kinds in one array are well-formed (spec §4): mortie's
        # kernel takes one order per call, so points (clipped to 24) and
        # areas fabricate separately and reassemble by position.
        words = [AREA29_NORTH_WORD, POINT_NORTH_WORD, POINT_SOUTH_WORD, AREA29_SOUTH_WORD]
        with pytest.warns(UserWarning, match="float64"):  # the area half
            ids = fabricate_cell_ids(words)
        assert ids.tolist() == [NESTED29_NORTH, NESTED24_NORTH, NESTED24_SOUTH, NESTED29_SOUTH]

    def test_point_level_cross_check(self):
        # level checks the ENCODED order: points are order-29 encodings even
        # though their NESTED view is the order-24 clip.
        ids = fabricate_cell_ids([POINT_NORTH_WORD], level=29)
        assert ids.tolist() == [NESTED24_NORTH]
        with pytest.raises(ValueError, match="order 29"):
            fabricate_cell_ids([POINT_NORTH_WORD], level=24)

    def test_golden_parity_serc(self):
        # The money test: fabricated ids from the (morton-only) fixture's
        # morton coordinate EXACTLY equal the cell_ids array the final
        # dual-written zagg fixture stored (frozen in the .npy golden).
        ds = open_hive(str(FIXTURE), fabricate_cell_ids=False)
        assert "cell_ids" not in ds.variables  # post-#314 stores are morton-only
        fabricated = fabricate_cell_ids(np.asarray(ds["morton"].values, dtype=np.uint64), level=8)
        assert fabricated.dtype == np.uint64
        np.testing.assert_array_equal(fabricated, golden_cell_ids())


class TestOpenHiveFabrication:
    def test_auto_keeps_stored(self, tmp_path):
        # Default ("auto") on a store that carries cell_ids (the zagg
        # emit_cell_ids transition hatch): the stored coordinate rides
        # through untouched — the deviant bytes prove no refabrication.
        ds_auto = open_hive(_dual_written_copy(tmp_path))
        assert "cell_ids" in ds_auto.coords
        np.testing.assert_array_equal(ds_auto["cell_ids"].values, golden_cell_ids() + np.uint64(1))

    def test_auto_fabricates_when_absent(self):
        # The fixture is morton-only (zagg#314): "auto" fabricates NESTED.
        ds = open_hive(str(FIXTURE))
        assert "cell_ids" in ds.coords
        assert ds["cell_ids"].dtype == np.uint64
        np.testing.assert_array_equal(ds["cell_ids"].values, golden_cell_ids())

    def test_false_absent_stays_absent(self):
        ds = open_hive(str(FIXTURE), fabricate_cell_ids=False)
        assert "cell_ids" not in ds.variables

    def test_true_forces_fabrication(self, tmp_path):
        # True fabricates even when a stored array exists — the deviant
        # stored bytes are REPLACED by the exact fabrication.
        ds = open_hive(_dual_written_copy(tmp_path), fabricate_cell_ids=True)
        np.testing.assert_array_equal(ds["cell_ids"].values, golden_cell_ids())

    def test_invalid_posture_raises(self):
        with pytest.raises(ValueError, match="fabricate_cell_ids"):
            open_hive(str(FIXTURE), fabricate_cell_ids="always")

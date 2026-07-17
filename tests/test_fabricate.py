"""``fabricate_cell_ids``: exact NESTED views from packed morton words.

The money test is golden parity: fabricated ids for the SERC fixture's
``morton`` coordinate byte-for-byte equal the fixture's STORED ``cell_ids``
array — zagg-written ground truth, so equality proves the morton-only
decision (englacial/zagg#262, "NESTED is fabricated, never stored") loses
nothing. The opener tests simulate a post-flip morton-only store by deleting
the ``cell_ids`` array from each leaf of a fixture copy.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from moczarr import open_hive, store
from moczarr.fabricate import FLOAT64_EXACT_MAX_ORDER, fabricate_cell_ids

FIXTURE = Path(__file__).parent / "data" / "serc_hive"
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


def _morton_only_copy(tmp_path):
    """A fixture copy with every leaf's ``cell_ids`` array deleted."""
    copy = tmp_path / "serc_morton_only"
    shutil.copytree(FIXTURE, copy)
    group = str(store.read_manifest(str(copy))["cell_order"])
    for rel in store.walk_leaves(str(copy)):
        cell_ids_dir = copy / rel / group / "cell_ids"
        if cell_ids_dir.exists():
            shutil.rmtree(cell_ids_dir)
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

    def test_golden_parity_serc(self):
        # The money test: fabricated ids from the fixture's morton coordinate
        # EXACTLY equal the zagg-written stored cell_ids array.
        ds = open_hive(str(FIXTURE), fabricate_cell_ids=False)
        stored = np.asarray(ds["cell_ids"].values, dtype=np.uint64)
        fabricated = fabricate_cell_ids(np.asarray(ds["morton"].values, dtype=np.uint64), level=8)
        assert fabricated.dtype == stored.dtype == np.uint64
        np.testing.assert_array_equal(fabricated, stored)


class TestOpenHiveFabrication:
    def test_auto_keeps_stored(self):
        # Default ("auto") on a store that carries cell_ids: the stored
        # coordinate rides through untouched — identical to a no-fabrication
        # open.
        ds_auto = open_hive(str(FIXTURE))
        ds_plain = open_hive(str(FIXTURE), fabricate_cell_ids=False)
        assert "cell_ids" in ds_auto.coords
        np.testing.assert_array_equal(ds_auto["cell_ids"].values, ds_plain["cell_ids"].values)

    def test_auto_fabricates_when_absent(self, tmp_path):
        golden = open_hive(str(FIXTURE), fabricate_cell_ids=False)["cell_ids"].values
        ds = open_hive(_morton_only_copy(tmp_path))
        assert "cell_ids" in ds.coords
        assert ds["cell_ids"].dtype == np.uint64
        np.testing.assert_array_equal(ds["cell_ids"].values, golden)

    def test_false_absent_stays_absent(self, tmp_path):
        ds = open_hive(_morton_only_copy(tmp_path), fabricate_cell_ids=False)
        assert "cell_ids" not in ds.variables

    def test_true_forces_fabrication(self):
        # True fabricates even when a stored array exists; parity means the
        # replacement equals the stored bytes.
        ds = open_hive(str(FIXTURE), fabricate_cell_ids=True)
        stored = open_hive(str(FIXTURE), fabricate_cell_ids=False)["cell_ids"].values
        np.testing.assert_array_equal(ds["cell_ids"].values, stored)

    def test_invalid_posture_raises(self):
        with pytest.raises(ValueError, match="fabricate_cell_ids"):
            open_hive(str(FIXTURE), fabricate_cell_ids="always")

"""The derived cell coordinate at the leaf-open seams (issue #71).

Zagg spec §1.5 ("The cell coordinate"): a windowed leaf
(``{id}_{window}.zarr``) stores no per-cell ``morton`` array — the words are
a pure function of the leaf's id and the rank — while an unwindowed leaf
still stores it. Readers carry one rule: use the stored array where the leaf
has it, derive it where a windowed leaf does not; absence on an unwindowed
leaf is corruption.

Every parity claim here is "the same leaf with and without its stored
coordinate reads the same": the committed fixtures (whose leaves all store
the array) are **copied and stripped** by ``conftest.strip_morton``, and the
stripped copy is read against the original. The zagg-written morton-less
conformance fixture is exercised in ``test_spec_conformance.py``.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from conftest import strip_morton
from test_ragged import CountingStore
from zarr.storage import LocalStore

from moczarr import convention, open_hive, open_leaf, store
from moczarr.hhdc import cell_index, has_exact_occupancy, read_tensors
from moczarr.ragged import _morton_words, read_ragged

DATA = Path(__file__).parent / "data"
WINDOWS = DATA / "multiproduct_hive" / "atl06_windows"
SERC = DATA / "serc_hive"
STRATA = json.loads((DATA / "strata_hive.expected.json").read_text())
STRATA_LEAF = DATA / "strata_hive" / STRATA["leaf"]
GROUP = STRATA["group"]
SIGNAL = f"{GROUP}/h_tdigest_signal"
NOISE = f"{GROUP}/h_tdigest_noise"
#: The strata fixture's read chunks (4 cells each): 433141/2/4 hold data,
#: 433143 is the unwritten one.
UNWRITTEN_CHUNK = "433143"


def _windowed_leaves(root, window):
    return [
        rel
        for rel in sorted(store.walk_leaves(str(root)))
        if convention.split_leaf_name(rel.rsplit("/", 1)[-1])[1] == window
    ]


@pytest.fixture()
def derived_windows(tmp_path):
    """The windowed fixture with EVERY 2019 leaf's ``morton`` array removed."""
    root = tmp_path / "windows"
    shutil.copytree(WINDOWS, root)
    for rel in _windowed_leaves(root, "2019"):
        strip_morton(root / rel, 5)
    return str(root)


@pytest.fixture()
def mixed_windows(tmp_path):
    """One 2019 leaf derived, the other still storing its array — the store
    §1.5 says stays valid as it stands (pre-revision windowed leaves)."""
    root = tmp_path / "windows"
    shutil.copytree(WINDOWS, root)
    strip_morton(root / _windowed_leaves(root, "2019")[0], 5)
    return str(root)


@pytest.fixture()
def derived_strata(tmp_path):
    """The strata leaf as a windowed leaf with no stored coordinate, as a
    leaf-rooted store — what the per-leaf readers are handed."""
    root = tmp_path / "leaf"
    shutil.copytree(STRATA_LEAF, root)
    strip_morton(root, GROUP, window="2019")
    return root


class TestOpenHive:
    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    @pytest.mark.parametrize("fixture", ["derived_windows", "mixed_windows"])
    def test_dataset_is_identical_to_the_stored_coordinate_twin(self, fixture, index_kind, request):
        root = request.getfixturevalue(fixture)
        stored = open_hive(str(WINDOWS), window="2019", index_kind=index_kind).load()
        derived = open_hive(root, window="2019", index_kind=index_kind).load()
        assert "morton" in derived.coords and "cell_ids" in derived.coords
        assert derived["morton"].dtype == np.uint64
        xr.testing.assert_identical(derived, stored)

    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    def test_cell_exact_aoi_rows_come_from_the_derived_words(self, derived_windows, index_kind):
        # Tier 2 — "rows exactly, via the morton coordinate" — on a leaf that
        # stores none: three cells of one shard, by word.
        words = convention.leaf_cell_words("-5111", 5)[[1, 6, 15]]
        stored = open_hive(str(WINDOWS), window="2019", aoi=words, index_kind=index_kind).load()
        derived = open_hive(derived_windows, window="2019", aoi=words, index_kind=index_kind).load()
        assert derived.sizes["cells"] == 3
        np.testing.assert_array_equal(derived["morton"].values, words)
        xr.testing.assert_identical(derived, stored)

    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    def test_empty_aoi_schema_carries_the_coordinate(self, derived_windows, index_kind):
        outside = [convention.morton_word("1111")]
        with pytest.warns(UserWarning, match="intersects no coverage"):
            stored = open_hive(str(WINDOWS), window="2019", aoi=outside, index_kind=index_kind)
        with pytest.warns(UserWarning, match="intersects no coverage"):
            derived = open_hive(derived_windows, window="2019", aoi=outside, index_kind=index_kind)
        assert derived.sizes["cells"] == 0 and "morton" in derived.coords
        xr.testing.assert_identical(derived, stored)

    def test_decode_assigns_the_same_xdggs_index(self, derived_windows):
        pytest.importorskip("xdggs")
        stored = open_hive(str(WINDOWS), window="2019", decode=True, index_kind="pandas")
        derived = open_hive(derived_windows, window="2019", decode=True, index_kind="pandas")
        xr.testing.assert_identical(derived.load(), stored.load())
        assert type(derived.xindexes["morton"]) is type(stored.xindexes["morton"])

    def test_the_other_window_is_untouched(self, derived_windows):
        # 2020's leaf still stores its array: one store, both kinds of leaf.
        stored = open_hive(str(WINDOWS), window="2020", index_kind="pandas").load()
        xr.testing.assert_identical(
            open_hive(derived_windows, window="2020", index_kind="pandas").load(), stored
        )

    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    def test_an_unwindowed_leaf_without_the_array_is_refused(self, tmp_path, index_kind):
        # §1.5: "Absence on an unwindowed leaf is corruption, not a licence
        # to derive, and MUST be refused" — on the lazy path too, which
        # never reads the array and would otherwise not notice.
        root = tmp_path / "serc"
        shutil.copytree(SERC, root)
        strip_morton(root / convention.leaf_path("4331422"), 8)
        with pytest.raises(ValueError, match="only a windowed leaf"):
            open_hive(str(root), index_kind=index_kind)

    @pytest.mark.parametrize("drop", [["morton"], "morton"])
    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    def test_a_caller_dropped_coordinate_is_neither_refused_nor_derived(
        self, derived_windows, index_kind, drop
    ):
        # Dropping ``morton`` at open is the caller's choice, not a store
        # without the array: an unwindowed leaf still opens, and a derived
        # windowed one comes back without the coordinate, as asked.
        kwargs = {"index_kind": index_kind, "fabricate_cell_ids": False}
        serc = open_hive(str(SERC), xr_kwargs={"drop_variables": drop}, **kwargs)
        assert serc.sizes["cells"] == open_hive(str(SERC), **kwargs).sizes["cells"]
        derived = open_hive(
            derived_windows, window="2019", xr_kwargs={"drop_variables": drop}, **kwargs
        )
        assert derived.sizes["cells"] == 32
        if index_kind == "pandas":
            assert "morton" not in serc.variables and "morton" not in derived.variables

    def test_a_cells_axis_that_is_not_the_shard_subtree_is_refused(self, derived_windows):
        # The derivation is the shard's children — all of them. A leaf whose
        # axis is some other length has no coordinate to derive.
        rel = _windowed_leaves(derived_windows, "2019")[0]
        meta_path = Path(derived_windows) / rel / "5" / "height" / "zarr.json"
        meta = json.loads(meta_path.read_text())
        meta["shape"] = [12]
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="exactly its shard's subtree"):
            open_hive(derived_windows, window="2019", index_kind="pandas")

    def test_no_request_names_a_morton_array(self, derived_windows, monkeypatch):
        import obstore

        seen = []
        real = obstore.get_async

        def spy(handle, path, *args, **kwargs):
            seen.append(path)
            return real(handle, path, *args, **kwargs)

        monkeypatch.setattr(obstore, "get_async", spy)
        open_hive(derived_windows, window="2019", index_kind="pandas").load()
        assert seen and not [key for key in seen if "/morton" in key]


class TestRaggedReaders:
    """The store-rooted readers: the leaf's id comes off its own stamp (the
    coverage box names the shard), and "windowed" off the stamp's ``window``."""

    def test_derived_words_are_the_stored_array_made_full(self, derived_strata):
        stored = np.asarray(_morton_words(LocalStore(STRATA_LEAF), SIGNAL, 3)[0:16])
        derived = _morton_words(LocalStore(derived_strata), SIGNAL, 3)
        assert isinstance(derived, np.ndarray) and derived.dtype == np.uint64
        np.testing.assert_array_equal(
            derived, convention.leaf_cell_words(STRATA["shard"], int(GROUP))
        )
        written = stored != 0
        assert 0 < written.sum() < 16  # the stored twin is sparse: fill on one chunk
        np.testing.assert_array_equal(derived[written], stored[written])
        assert np.all(derived != 0)  # full by construction

    @pytest.mark.parametrize("field", [SIGNAL, NOISE])
    def test_read_ragged_yields_the_same_cells(self, derived_strata, field):
        stored = list(read_ragged(LocalStore(STRATA_LEAF), field, locations=True))
        derived = list(read_ragged(LocalStore(derived_strata), field, locations=True))
        assert len(derived) == len(stored) > 0
        for (word, values, locations), (s_word, s_values, s_locations) in zip(derived, stored):
            assert word == s_word
            np.testing.assert_array_equal(values, s_values)
            np.testing.assert_array_equal(locations, s_locations)

    def test_read_ragged_subtree_resolves_against_the_derived_words(self, derived_strata):
        for subtree in ("433141", "433144", UNWRITTEN_CHUNK, STRATA["shard"]):
            stored = [w for w, _v in read_ragged(LocalStore(STRATA_LEAF), SIGNAL, subtree=subtree)]
            derived = [
                w for w, _v in read_ragged(LocalStore(derived_strata), SIGNAL, subtree=subtree)
            ]
            assert derived == stored

    @pytest.mark.parametrize("block_order", [None, 4])
    @pytest.mark.parametrize("field", [SIGNAL, NOISE])
    def test_read_tensors_blocks_are_identical(self, derived_strata, field, block_order):
        stored = list(read_tensors(LocalStore(STRATA_LEAF), field, block_order=block_order))
        derived = list(read_tensors(LocalStore(derived_strata), field, block_order=block_order))
        assert len(derived) == len(stored) > 0
        for (tensor, mask, window, word), (s_tensor, s_mask, s_window, s_word) in zip(
            derived, stored
        ):
            np.testing.assert_array_equal(tensor, s_tensor)
            np.testing.assert_array_equal(mask, s_mask)  # occupancy is the sidecar's
            assert window == s_window and word == s_word

    def test_no_read_touches_a_morton_key(self, derived_strata):
        counting = CountingStore(derived_strata)
        assert list(read_tensors(counting, SIGNAL))
        assert counting.gets and not [g for g in counting.gets if "morton" in g[0]]

    def test_open_leaf_feeds_the_readers(self, derived_windows):
        # The leaf-direct door on a real windowed store: the store it returns
        # carries the stamp the derivation reads.
        manifest = store.read_manifest(derived_windows)
        derived = _morton_words(open_leaf(derived_windows, "-5111", window="2019"), "5/height", 3)
        stored = _morton_words(open_leaf(str(WINDOWS), "-5111", window="2019"), "5/height", 3)
        assert int(manifest["cell_order"]) == 5
        np.testing.assert_array_equal(derived, np.asarray(stored[0:16]))


class TestOccupancy:
    """A derived coordinate is full by construction, so "was this written?"
    is never its question to answer (§1.5: occupancy from the payload arrays
    or the stamp's coverage)."""

    def test_cell_index_resolves_every_written_cell(self, derived_strata):
        # The stored twin answers this off its coordinate's fill; the derived
        # leaf off the stamp's exact coverage — cell for cell the same,
        # including cells whose SIGNAL stratum is empty in a written chunk.
        stored, derived = LocalStore(STRATA_LEAF), LocalStore(derived_strata)
        assert has_exact_occupancy(derived)
        for cell in STRATA["cells"]:
            chunk = convention.morton_decimal(int(cell["morton"]))[:-1]
            rank = cell["index"] % 4
            row, col = rank >> 1, rank & 1  # only the composition matters here
            assert cell_index(derived, SIGNAL, chunk, row, col) == cell_index(
                stored, SIGNAL, chunk, row, col
            )

    def test_cell_index_refuses_the_unwritten_chunk(self, derived_strata):
        # Its derived words are perfectly good words — which is exactly why
        # the coordinate cannot be what says nothing was written there.
        words = _morton_words(LocalStore(derived_strata), SIGNAL, 3)[8:12]
        assert np.all(words != 0)
        for leaf in (STRATA_LEAF, derived_strata):
            with pytest.raises(ValueError, match="no stored read chunk"):
                cell_index(LocalStore(leaf), SIGNAL, UNWRITTEN_CHUNK, 0, 0)

    def test_cell_index_reads_no_digest_bytes_with_exact_coverage(self, derived_strata):
        counting = CountingStore(derived_strata)
        assert cell_index(counting, SIGNAL, "433141", 1, 0) == 2
        assert [g for g in counting.gets if f"{SIGNAL}/c/" in g[0]] == []
        assert [g for g in counting.gets if g[0] == "coverage.moc"]

    def test_cell_index_falls_back_to_the_payload_without_exact_coverage(self, derived_strata):
        # A box-only stamp: no cell-exact record, so the chunk itself is
        # read — the cell-order group's payload arrays, any of them.
        _box_only(derived_strata)
        leaf = LocalStore(derived_strata)
        assert not has_exact_occupancy(leaf)
        assert cell_index(leaf, SIGNAL, "433141", 1, 0) == 2
        with pytest.raises(ValueError, match="no stored read chunk"):
            cell_index(leaf, SIGNAL, UNWRITTEN_CHUNK, 0, 0)

    @pytest.mark.parametrize("box_only", [False, True])
    @pytest.mark.parametrize("hive_rooted", [False, True])
    def test_a_chunk_another_field_wrote_resolves_as_on_the_stored_twin(
        self, tmp_path, hive_rooted, box_only
    ):
        # Read chunk 433142 holds NOISE at cell 5 and no SIGNAL at all; the
        # stored twin's coordinate is written there, so SIGNAL resolves —
        # whether the stamp is exact or box-only, and whether the store is
        # rooted at the leaf or at the hive (the stamp is the LEAF's, found
        # off the field path, never the store root's).
        root = tmp_path / "hive"
        shutil.copytree(DATA / "strata_hive", root)
        prefix = f"{STRATA['leaf']}/" if hive_rooted else ""
        twin = LocalStore(DATA / "strata_hive" if hive_rooted else STRATA_LEAF)
        stored = cell_index(twin, prefix + SIGNAL, "433142", 0, 1)
        assert stored == 5
        strip_morton(root / STRATA["leaf"], GROUP, window="2019")
        if box_only:
            _box_only(root / STRATA["leaf"])
        derived = LocalStore(root if hive_rooted else root / STRATA["leaf"])
        assert cell_index(derived, prefix + SIGNAL, "433142", 0, 1) == stored
        with pytest.raises(ValueError, match="no stored read chunk"):
            cell_index(derived, prefix + SIGNAL, UNWRITTEN_CHUNK, 0, 0)

    def test_a_hive_rooted_call_reads_no_digest_bytes_with_exact_coverage(self, tmp_path):
        root = tmp_path / "hive"
        shutil.copytree(DATA / "strata_hive", root)
        strip_morton(root / STRATA["leaf"], GROUP, window="2019")
        counting = CountingStore(root)
        field = f"{STRATA['leaf']}/{SIGNAL}"
        assert cell_index(counting, field, "433141", 1, 0) == 2
        assert [g for g in counting.gets if f"{field}/c/" in g[0]] == []
        assert [g for g in counting.gets if g[0] == f"{STRATA['leaf']}/coverage.moc"]


def _box_only(leaf_dir):
    """Drop a copied leaf stamp's exact coverage, keeping its tier-0 box."""
    meta_path = Path(leaf_dir) / "zarr.json"
    meta = json.loads(meta_path.read_text())
    coverage = meta["attributes"][convention.COMMIT_ATTR]["coverage"]
    for key in ("encoding", "sidecar", "nbytes", "raw_nbytes"):
        coverage.pop(key)
    meta_path.write_text(json.dumps(meta))


class TestRefusals:
    """Where the array's absence is still an error."""

    def test_an_unwindowed_leaf_without_the_array(self, tmp_path):
        root = tmp_path / "leaf"
        shutil.copytree(STRATA_LEAF, root)
        strip_morton(root, GROUP)  # the stamp names no window
        with pytest.raises(ValueError, match="no sibling 'morton'.*only a windowed leaf"):
            list(read_ragged(LocalStore(root), SIGNAL))
        with pytest.raises(ValueError, match="no sibling 'morton'"):
            list(read_tensors(LocalStore(root), SIGNAL))

    def test_an_unstamped_store(self, tmp_path):
        root = tmp_path / "leaf"
        shutil.copytree(STRATA_LEAF, root)
        strip_morton(root, GROUP, window="2019")
        meta = json.loads((root / "zarr.json").read_text())
        meta["attributes"].pop(convention.COMMIT_ATTR)
        (root / "zarr.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="no sibling 'morton'"):
            list(read_ragged(LocalStore(root), SIGNAL))

    def test_a_column_is_not_a_leaf(self, tmp_path):
        # §1.5: the §4 artifacts store their coordinate at every revision. A
        # windowed column's stamp names ``window`` too, so the role attr —
        # not the stamp — is what keeps a derivation off it.
        root = tmp_path / "column"
        shutil.copytree(DATA / "spec" / "temporal" / "1/1/2/1/3/all.pyramid.zarr", root)
        strip_morton(root, 5, window="2019")
        assert json.loads((root / "zarr.json").read_text())["attributes"]["role"] == "column"
        with pytest.raises(ValueError, match="no sibling 'morton'"):
            list(read_ragged(LocalStore(root), "5/h_tdigest"))

    def test_a_group_that_is_not_the_cell_order_group(self, derived_strata):
        (derived_strata / GROUP).rename(derived_strata / "9")
        with pytest.raises(ValueError, match="no sibling 'morton'"):
            list(read_ragged(LocalStore(derived_strata), "9/h_tdigest_signal"))

    @pytest.mark.parametrize("box", [[None, None, None, None], None])
    def test_a_windowed_leaf_whose_stamp_names_no_shard(self, derived_strata, box):
        meta_path = derived_strata / "zarr.json"
        meta = json.loads(meta_path.read_text())
        coverage = meta["attributes"][convention.COMMIT_ATTR]["coverage"]
        if box is None:
            del coverage["box"]
        else:
            coverage["box"] = box
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="cannot identify the shard"):
            list(read_ragged(LocalStore(derived_strata), SIGNAL))

    def test_a_windowed_leaf_with_no_coverage_envelope(self, derived_strata):
        meta_path = derived_strata / "zarr.json"
        meta = json.loads(meta_path.read_text())
        del meta["attributes"][convention.COMMIT_ATTR]["coverage"]
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="cannot identify the shard"):
            list(read_ragged(LocalStore(derived_strata), SIGNAL))

    def test_a_box_member_coarser_than_the_shard(self, derived_strata):
        meta_path = derived_strata / "zarr.json"
        meta = json.loads(meta_path.read_text())
        meta["attributes"][convention.COMMIT_ATTR]["coverage"]["box"] = ["433", None, None, None]
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="coarser than the order-4 shard"):
            list(read_ragged(LocalStore(derived_strata), SIGNAL))

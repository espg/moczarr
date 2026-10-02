"""Versioned leaves: following the ``current`` pointer (issue #70).

Zagg spec §1.5 ("Versioned leaves"): a leaf's stable ``{id}.zarr/`` prefix
may be a **pointer root** — its root stamp names ``current``, the version
subgroup holding the arrays and the in-leaf ``coverage.moc``. The reader
carries one rule (open the root; follow ``current`` when named, else read
the root), and it costs no request, because the stamp it reads anyway is
the pointer.

The stores here are the committed zagg-written fixtures, **copied and
rewritten** into the versioned shape by ``conftest.version_leaf`` — so every
parity claim is "the same bytes behind a pointer read the same", against a
legacy original that the rest of the suite already pins. The zagg-written
versioned conformance fixture is exercised in ``test_spec_conformance.py``.
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from conftest import DEBRIS, STAMPED, build_many_leaf_store, version_leaf

from moczarr import (
    convention,
    hash_arrays,
    occupancy_and,
    open_hive,
    open_leaf,
    store,
    verify_arrays,
)
from moczarr.store import leaf_data_prefix

DATA = Path(__file__).parent / "data"
SERC = DATA / "serc_hive"
SERC_SHARD = "4331422"
ATL06 = DATA / "multiproduct_hive" / "atl06"
WINDOWS = DATA / "multiproduct_hive" / "atl06_windows"
LEAF = "4/3/3/1/4/2/2/4331422.zarr"


def _stamped_leaves(root):
    return [rel for rel in sorted(store.walk_leaves(root)) if store.read_commit(root, rel)]


@pytest.fixture()
def versioned_serc(tmp_path):
    """The SERC fixture with EVERY stamped leaf rewritten as a versioned leaf."""
    root = tmp_path / "serc"
    shutil.copytree(SERC, root)
    for rel in _stamped_leaves(str(root)):
        version_leaf(root, rel)
    return str(root)


@pytest.fixture()
def mixed_serc(tmp_path):
    """The SERC fixture with every OTHER stamped leaf versioned — the store
    §1.5 says is legitimate after its first post-revision write."""
    root = tmp_path / "serc"
    shutil.copytree(SERC, root)
    for rel in _stamped_leaves(str(root))[::2]:
        version_leaf(root, rel)
    return str(root)


class TestLeafDataPrefix:
    def test_legacy_stamp_is_the_root(self):
        assert leaf_data_prefix(LEAF, {"spec": "morton-hive/1", "complete": True}) == LEAF

    def test_debris_is_the_root(self):
        # No stamp, no pointer: the root, which holds nothing a reader trusts.
        assert leaf_data_prefix(LEAF, None) == LEAF

    def test_current_names_the_version_subgroup(self):
        assert leaf_data_prefix(LEAF, {"current": "run-abc-0a1b2c3d"}) == f"{LEAF}/run-abc-0a1b2c3d"

    def test_slashes_are_normalized_like_every_other_leaf_key(self):
        assert leaf_data_prefix(f"/{LEAF}/", {"current": "run-a-b"}) == f"{LEAF}/run-a-b"
        assert leaf_data_prefix(f"/{LEAF}/", None) == LEAF

    @pytest.mark.parametrize(
        "current",
        [
            "v1",  # not a version name
            "8",  # a cell-order digit group is never a version (§4.2)
            "run-abc/8",  # more than one path component
            "../run-abc",  # a traversal, and not a version name either
            "",  # names nothing
            7,  # not a string
            True,
            ["run-abc"],
        ],
    )
    def test_invalid_current_is_refused(self, current):
        with pytest.raises(ValueError, match="invalid version"):
            leaf_data_prefix(LEAF, {"current": current})

    def test_explicit_null_reads_as_absent(self):
        # JSON null is "no value", the same answer as the key being absent.
        assert leaf_data_prefix(LEAF, {"current": None}) == LEAF


class TestFixtureShape:
    """What ``version_leaf`` builds is the §1.5 shape — the premise of every
    parity test below."""

    def test_root_holds_only_the_pointer_and_its_version(self, versioned_serc):
        rel = convention.leaf_path(SERC_SHARD)
        stamp = store.read_commit(versioned_serc, rel)
        root = Path(versioned_serc) / rel
        assert sorted(p.name for p in root.iterdir()) == sorted(["zarr.json", stamp["current"]])
        version = store.read_commit(versioned_serc, f"{rel}/{stamp['current']}")
        assert "current" not in version
        assert version == {k: v for k, v in stamp.items() if k != "current"}

    def test_the_walk_sees_the_same_leaves(self, versioned_serc):
        # §4.2: versions are subgroups of the leaf entry, so the node's
        # children — what the discovery walk classifies — are unchanged.
        assert sorted(store.walk_leaves(versioned_serc)) == sorted(store.walk_leaves(str(SERC)))


class TestOpenHive:
    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    @pytest.mark.parametrize("fixture", ["versioned_serc", "mixed_serc"])
    def test_whole_store_is_identical_to_legacy(self, fixture, index_kind, request):
        root = request.getfixturevalue(fixture)
        legacy = open_hive(str(SERC), index_kind=index_kind).load()
        versioned = open_hive(root, index_kind=index_kind).load()
        xr.testing.assert_identical(versioned, legacy)

    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    def test_cell_exact_aoi_is_identical_to_legacy(self, versioned_serc, index_kind):
        occupied = store.read_coverage_bitmap(str(SERC), convention.leaf_path(SERC_SHARD))
        legacy = open_hive(str(SERC), aoi=occupied, index_kind=index_kind).load()
        versioned = open_hive(versioned_serc, aoi=occupied, index_kind=index_kind).load()
        assert legacy.sizes["cells"] == len(occupied) > 0
        xr.testing.assert_identical(versioned, legacy)

    def test_empty_aoi_takes_its_schema_from_a_versioned_leaf(self, versioned_serc):
        # The schema-correct empty return (issue #4) reads ONE leaf's
        # metadata — through the pointer, like any other open.
        outside = [convention.morton_word("1111111")]
        with pytest.warns(UserWarning, match="intersects no coverage"):
            legacy = open_hive(str(SERC), aoi=outside)
        with pytest.warns(UserWarning, match="intersects no coverage"):
            versioned = open_hive(versioned_serc, aoi=outside)
        assert versioned.sizes["cells"] == 0
        xr.testing.assert_identical(versioned, legacy)

    def test_empty_walk_schema_comes_through_the_pointer(self, versioned_serc):
        # No root MOC: the schema leaf is found by the walk, and its stamp —
        # not the candidates' — is what locates its arrays.
        (Path(versioned_serc) / convention.ROOT_COVERAGE_NAME).unlink()
        outside = [convention.morton_word("1111111")]
        with pytest.warns(UserWarning, match="intersects no coverage"):
            ds = open_hive(versioned_serc, aoi=outside)
        assert ds.sizes["cells"] == 0 and "count" in ds

    def test_converted_legacy_leaf_reads_the_version_not_the_root(self, tmp_path):
        # §1.5: a pointer over a root that ALSO holds {cell_order}/… arrays is
        # a converted legacy leaf; the root arrays are the last legacy write
        # and a reader following ``current`` never reads them.
        shard = "-5112333"
        root = build_many_leaf_store(tmp_path / "store", [shard])
        rel = convention.leaf_path(shard)
        version = version_leaf(root, rel, keep_root=True)
        stale = Path(root) / rel / "8" / "count" / "c" / "0"
        fresh = Path(root) / rel / version / "8" / "count" / "c" / "0"
        stale.write_bytes(np.full(16, 7, dtype="<i8").tobytes())
        fresh.write_bytes(np.full(16, 3, dtype="<i8").tobytes())
        assert set(open_hive(root)["count"].values.tolist()) == {3}

    def test_pointer_costs_no_extra_request(self, versioned_serc, monkeypatch):
        # The stamp read IS the pointer read: a versioned store is opened
        # with exactly the legacy store's requests, each leaf-interior key
        # re-prefixed by the version and nothing added — in particular no
        # GET of the version's own zarr.json.
        legacy = _requests(monkeypatch, lambda: open_hive(str(SERC)).load())
        versioned = _requests(monkeypatch, lambda: open_hive(versioned_serc).load())
        version = store.read_commit(versioned_serc, convention.leaf_path(SERC_SHARD))["current"]
        assert not any(key.endswith(f"{version}/zarr.json") for _fn, key in versioned)
        assert sorted(versioned) == sorted((fn, _reprefixed(key, version)) for fn, key in legacy)


def _reprefixed(key, version):
    """A legacy request key as the versioned store names it."""
    head, sep, tail = key.partition(".zarr/")
    if not sep or tail == "zarr.json":
        return key  # not under a leaf, or the stable root's own stamp
    return f"{head}{sep}{version}/{tail}"


def _requests(monkeypatch, call):
    """Every obstore request ``call`` issues, as ``(function, key)`` pairs."""
    import obstore

    seen = []

    def spy(name):
        real = getattr(obstore, name)

        def wrapper(handle, *args, **kwargs):
            key = args[0] if args else kwargs.get("prefix", kwargs.get("path"))
            seen.append((name, (key or "").strip("/")))
            return real(handle, *args, **kwargs)

        return wrapper

    with monkeypatch.context() as patch:
        for name in (
            "get",
            "get_async",
            "get_range_async",
            "get_ranges_async",
            "head_async",
            "list",
            "list_with_delimiter",
            "list_with_delimiter_async",
        ):
            patch.setattr(obstore, name, spy(name))
        call()
    return seen


class TestCorruptedLeaf:
    """§1.5: a pointer naming a missing or unstamped version is a corrupted
    leaf that a reader treats as debris."""

    @pytest.fixture()
    def dangling(self, versioned_serc):
        rel = convention.leaf_path(SERC_SHARD)
        version = store.read_commit(versioned_serc, rel)["current"]
        shutil.rmtree(Path(versioned_serc) / rel / version)
        return versioned_serc, rel, version

    @pytest.mark.parametrize("index_kind", ["moc", "pandas"])
    def test_open_hive_skips_it_and_says_so(self, dangling, index_kind):
        root, _rel, version = dangling
        with pytest.warns(UserWarning, match=f"points at version '{version}'"):
            ds = open_hive(root, index_kind=index_kind).load()
        legacy = open_hive(str(SERC), index_kind=index_kind).load()
        lo = np.uint64(convention.morton_word(SERC_SHARD + "11"))
        hi = np.uint64(convention.morton_word(SERC_SHARD + "44"))
        outside = (legacy["morton"] < lo) | (legacy["morton"] > hi)
        assert ds.sizes["cells"] == legacy.sizes["cells"] - 16
        np.testing.assert_array_equal(ds["morton"].values, legacy["morton"].values[outside])
        np.testing.assert_array_equal(ds["count"].values, legacy["count"].values[outside])

    def test_unstamped_version_without_arrays_is_the_same_debris(self, versioned_serc):
        rel = convention.leaf_path(SERC_SHARD)
        version = store.read_commit(versioned_serc, rel)["current"]
        shutil.rmtree(Path(versioned_serc) / rel / version / "8")
        (Path(versioned_serc) / rel / version / "zarr.json").unlink()
        with pytest.warns(UserWarning, match="corrupted leaf"):
            ds = open_hive(versioned_serc)
        assert ds.sizes["cells"] == open_hive(str(SERC)).sizes["cells"] - 16

    def test_unstamped_version_with_its_arrays_is_served_by_open_hive(self, versioned_serc):
        # The deliberate gap (issue #70 review): the open never reads the
        # version's own stamp — catching this state would cost one GET per
        # versioned leaf — so the arrays are served, silently, while the
        # verifier, which reads that object anyway, reports debris.
        rel = convention.leaf_path(SERC_SHARD)
        version = store.read_commit(versioned_serc, rel)["current"]
        (Path(versioned_serc) / rel / version / "zarr.json").unlink()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds = open_hive(versioned_serc).load()
        xr.testing.assert_identical(ds, open_hive(str(SERC)).load())
        assert hash_arrays(versioned_serc, rel) == {}

    def test_stamped_version_missing_its_group_still_raises(self, versioned_serc):
        # NOT debris: the version is there and stamped, so a missing
        # cell-order group is a malformed leaf — loud, as on a legacy leaf.
        rel = convention.leaf_path(SERC_SHARD)
        version = store.read_commit(versioned_serc, rel)["current"]
        shutil.rmtree(Path(versioned_serc) / rel / version / "8")
        with pytest.raises(FileNotFoundError):
            open_hive(versioned_serc)

    def test_legacy_leaf_missing_its_group_still_raises(self, tmp_path):
        root = tmp_path / "serc"
        shutil.copytree(SERC, root)
        shutil.rmtree(root / convention.leaf_path(SERC_SHARD) / "8")
        with pytest.raises(FileNotFoundError):
            open_hive(str(root))

    def test_it_never_serves_the_empty_schema(self, dangling):
        # The schema source skips the corrupted leaf like any other debris:
        # an AOI over ONLY that leaf is empty, with a schema from elsewhere.
        root, _rel, _version = dangling
        with pytest.warns(UserWarning) as caught:
            ds = open_hive(root, aoi=[SERC_SHARD])
        assert ds.sizes["cells"] == 0 and "count" in ds
        messages = [str(w.message) for w in caught]
        assert any("corrupted leaf" in m for m in messages)
        assert any("intersects no coverage" in m for m in messages)

    def test_hash_arrays_reads_it_as_absent(self, dangling):
        root, rel, _version = dangling
        assert hash_arrays(root, rel) == {}

    def test_hash_arrays_refuses_an_unstamped_version(self, versioned_serc):
        # The arrays are all still there — only the version's stamp is gone.
        # The verifier reads that object anyway, so it holds the full rule.
        rel = convention.leaf_path(SERC_SHARD)
        version = store.read_commit(versioned_serc, rel)["current"]
        assert hash_arrays(versioned_serc, rel)
        (Path(versioned_serc) / rel / version / "zarr.json").unlink()
        assert hash_arrays(versioned_serc, rel) == {}

    def test_invalid_current_raises_on_every_open(self, versioned_serc):
        rel = convention.leaf_path(SERC_SHARD)
        path = Path(versioned_serc) / rel / "zarr.json"
        meta = json.loads(path.read_text())
        meta["attributes"][convention.COMMIT_ATTR]["current"] = "8"
        path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="invalid version '8'"):
            open_hive(versioned_serc)
        with pytest.raises(ValueError, match="invalid version '8'"):
            open_leaf(versioned_serc, SERC_SHARD)
        with pytest.raises(ValueError, match="invalid version '8'"):
            store.read_coverage_bitmap(versioned_serc, rel)
        with pytest.raises(ValueError, match="invalid version '8'"):
            hash_arrays(versioned_serc, rel)


class TestOpenLeaf:
    def test_store_is_rooted_at_the_version(self, versioned_serc):
        legacy = open_leaf(str(SERC), SERC_SHARD)
        versioned = open_leaf(versioned_serc, SERC_SHARD)
        for name in ("count", "morton"):
            np.testing.assert_array_equal(
                zarr.open_array(versioned, path=f"8/{name}", mode="r")[:],
                zarr.open_array(legacy, path=f"8/{name}", mode="r")[:],
            )
        # The version's own stamp and sidecar sit beside its arrays, so the
        # store-rooted readers (the HHDC occupancy channel) need no pointer.
        stamp = zarr.open_group(versioned, mode="r").attrs[convention.COMMIT_ATTR]
        assert "current" not in stamp and stamp["coverage"]["sidecar"] == "coverage.moc"

    def test_legacy_and_debris_root_at_the_stable_prefix(self, hive_store):
        for shard in (STAMPED, DEBRIS):
            opened = open_leaf(hive_store, shard)
            assert str(opened.store).rstrip('/")').endswith(f"{shard}.zarr")

    def test_threaded_manifest_and_handle_share_the_stamp_read(self, versioned_serc):
        manifest = store.read_manifest(versioned_serc)
        handle = store.open_object_store(versioned_serc)
        opened = open_leaf(versioned_serc, SERC_SHARD, manifest=manifest, store=handle)
        assert zarr.open_array(opened, path="8/count", mode="r").shape == (16,)


class TestCoverageSidecar:
    def test_bitmap_is_read_under_the_version(self, versioned_serc):
        rel = convention.leaf_path(SERC_SHARD)
        legacy = store.read_coverage_bitmap(str(SERC), rel)
        np.testing.assert_array_equal(store.read_coverage_bitmap(versioned_serc, rel), legacy)
        # ... with the stamp threaded (no stamp GET), and with the envelope
        # alone, which no longer suffices to locate the sidecar.
        stamp = store.read_commit(versioned_serc, rel)
        np.testing.assert_array_equal(
            store.read_coverage_bitmap(versioned_serc, rel, stamp=stamp), legacy
        )
        np.testing.assert_array_equal(
            store.read_coverage_bitmap(versioned_serc, rel, coverage=stamp["coverage"]), legacy
        )

    def test_a_stale_root_sidecar_is_never_read(self, tmp_path):
        # Converted legacy leaf: the root keeps the last legacy write's
        # sidecar. The version's is the one the pointer names.
        root = tmp_path / "serc"
        shutil.copytree(SERC, root)
        rel = convention.leaf_path(SERC_SHARD)
        version_leaf(root, rel, keep_root=True)
        expected = store.read_coverage_bitmap(str(SERC), rel)
        (root / rel / "coverage.moc").write_bytes(b"not a bitmap")
        np.testing.assert_array_equal(store.read_coverage_bitmap(str(root), rel), expected)

    def test_bitmap_and_follows_the_pointer(self, versioned_serc):
        rel = convention.leaf_path(SERC_SHARD)
        aoi = [convention.morton_word(SERC_SHARD)]
        np.testing.assert_array_equal(
            store.bitmap_and(versioned_serc, rel, aoi), store.bitmap_and(str(SERC), rel, aoi)
        )

    def test_two_store_intersection_is_identical_to_legacy(self, versioned_serc):
        legacy = occupancy_and(str(SERC), str(SERC))
        np.testing.assert_array_equal(occupancy_and(versioned_serc, str(SERC)), legacy)
        np.testing.assert_array_equal(occupancy_and(versioned_serc, versioned_serc), legacy)


class TestVerify:
    """The O11 verifier resolves the pointer first: hashes are keyed relative
    to the version root, the form §5.3 records (identical to a legacy leaf's)."""

    @pytest.fixture()
    def versioned_atl06(self, tmp_path):
        root = tmp_path / "atl06"
        shutil.copytree(ATL06, root)
        rel = convention.leaf_path("4111")
        return str(root), rel, version_leaf(root, rel)

    def test_verify_arrays_matches_the_sibling_sidecar(self, versioned_atl06):
        # The stats sidecar stays beside the STABLE root, unversioned.
        root, _rel, _version = versioned_atl06
        legacy = verify_arrays(str(ATL06), "4111")
        result = verify_arrays(root, "4111")
        assert result["match"] is True and result["combined_match"] is True
        assert result == legacy

    def test_hash_keys_are_relative_to_the_version(self, versioned_atl06):
        root, rel, version = versioned_atl06
        hashes = hash_arrays(root, rel)
        assert set(hashes) == {"5/morton", "5/count"}
        # A version addressed directly is a legacy-shaped prefix: same hashes.
        assert hash_arrays(root, f"{rel}/{version}") == hashes

    def test_superseded_versions_and_root_arrays_are_out_of_scope(self, tmp_path):
        # A converted leaf with a second, superseded version beside the
        # current one: only the CURRENT version's arrays are hashed.
        root = tmp_path / "atl06"
        shutil.copytree(ATL06, root)
        rel = convention.leaf_path("4111")
        version = version_leaf(root, rel, keep_root=True)
        shutil.copytree(root / rel / version, root / rel / "run-superseded-00000000")
        assert set(hash_arrays(str(root), rel)) == {"5/morton", "5/count"}
        assert verify_arrays(str(root), "4111")["match"] is True

    def test_an_unswapped_version_under_a_legacy_root_is_out_of_scope(self, tmp_path):
        # §1.5 write order: the version lands (1)-(2) before the pointer swap
        # (4), so a legacy root can hold a stamped ``run-…`` sibling of its
        # cell-order group — an attempt in flight, or one that died before
        # the swap. The path reader still sees the legacy leaf, unchanged.
        root = tmp_path / "atl06"
        shutil.copytree(ATL06, root)
        rel = convention.leaf_path("4111")
        legacy = hash_arrays(str(root), rel)
        version = root / rel / "run-deadbeef-00000000"
        shutil.copytree(root / rel / "5", version / "5")
        shutil.copy2(root / rel / "zarr.json", version / "zarr.json")
        assert "current" not in store.read_commit(str(root), rel)
        assert hash_arrays(str(root), rel) == legacy
        assert set(legacy) == {"5/morton", "5/count"}
        assert verify_arrays(str(root), "4111")["match"] is True

    def test_windowed_leaf(self, tmp_path):
        root = tmp_path / "windows"
        shutil.copytree(WINDOWS, root)
        version_leaf(root, convention.leaf_path("-5111", window="2019"))
        result = verify_arrays(str(root), "-5111", window="2019")
        assert result == verify_arrays(str(WINDOWS), "-5111", window="2019")
        assert result["match"] is True
        ds = open_hive(str(root), window="2019").load()
        xr.testing.assert_identical(ds, open_hive(str(WINDOWS), window="2019").load())

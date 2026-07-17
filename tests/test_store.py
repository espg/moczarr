"""Store layer over the local-fixture hive store (see ``conftest.py``)."""

import json

import numpy as np
import pytest
from conftest import DEBRIS, FULL, STAMPED

from moczarr import convention, store


def _words(*decimals):
    return np.asarray([convention.morton_word(d) for d in decimals], dtype=np.uint64)


class TestOpenObjectStore:
    def test_missing_local_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a readable store root"):
            store.open_object_store(str(tmp_path / "nope"))

    def test_s3_anonymous_constructs(self):
        s3 = store.open_object_store("s3://some-bucket/prefix", anonymous=True, region="us-west-2")
        assert type(s3).__name__ == "S3Store"


class TestManifest:
    def test_reads(self, hive_store):
        manifest = store.read_manifest(hive_store)
        assert manifest["spec"] == convention.HIVE_SPEC
        assert (manifest["cell_order"], manifest["shard_order"]) == (8, 6)

    def test_absent_reads_none(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert store.read_manifest(str(tmp_path / "empty")) is None

    def test_malformed_raises(self, hive_store):
        # The manifest is the bootstrap: garbage is an error, not a cache miss.
        import pathlib

        path = pathlib.Path(hive_store) / convention.MANIFEST_NAME
        path.write_text("not json {")
        with pytest.raises(json.JSONDecodeError):
            store.read_manifest(hive_store)
        path.write_text(json.dumps({"spec": "morton-hive/9"}))
        with pytest.raises(ValueError, match="unknown manifest spec"):
            store.read_manifest(hive_store)


class TestRootCoverage:
    def test_loads(self, hive_store):
        envelope = store.load_root_coverage(hive_store)
        assert envelope["encoding"] == "ranges"
        assert envelope["ranges"] == [[STAMPED, STAMPED]]

    def test_degrades_to_none(self, hive_store, tmp_path):
        import pathlib

        # Garbage JSON: tolerant (regenerable cache) -> absent, not an error.
        path = pathlib.Path(hive_store) / convention.ROOT_COVERAGE_NAME
        path.write_text("not json {")
        assert store.load_root_coverage(hive_store) is None
        # Unknown spec: same.
        path.write_text(json.dumps({"spec": "morton-moc/9", "encoding": "ranges"}))
        assert store.load_root_coverage(hive_store) is None
        # Absent: same.
        (tmp_path / "empty").mkdir()
        assert store.load_root_coverage(str(tmp_path / "empty")) is None


class TestCommitStamp:
    def test_stamped(self, hive_store):
        stamp = store.read_commit(hive_store, convention.leaf_path(STAMPED))
        assert stamp["complete"] is True
        assert stamp["coverage"]["encoding"] == "bitmap"

    def test_windowed_stamped(self, hive_store):
        stamp = store.read_commit(hive_store, convention.leaf_path(FULL, window="2019"))
        assert stamp["window"] == "2019"
        assert stamp["spec"] == convention.HIVE_SPEC_V2

    def test_debris_reads_none(self, hive_store):
        assert store.read_commit(hive_store, convention.leaf_path(DEBRIS)) is None

    def test_absent_leaf_reads_none(self, hive_store):
        # A never-written shard is a clean GET miss on every backend.
        assert store.read_commit(hive_store, convention.leaf_path("-5112332")) is None

    def test_corrupt_metadata_raises(self, hive_store):
        import pathlib

        leaf = convention.leaf_path(STAMPED)
        (pathlib.Path(hive_store) / leaf / "zarr.json").write_text("not json {")
        with pytest.raises(json.JSONDecodeError):
            store.read_commit(hive_store, leaf)


class TestBitmapReads:
    def test_bitmap_leaf(self, hive_store):
        words = store.read_coverage_bitmap(hive_store, convention.leaf_path(STAMPED))
        np.testing.assert_array_equal(words, _words(STAMPED + "11"))

    def test_full_leaf_reads_none(self, hive_store):
        # encoding "full": there IS no sidecar; bitmap_and short-circuits.
        leaf = convention.leaf_path(FULL, window="2019")
        assert store.read_coverage_bitmap(hive_store, leaf) is None

    def test_debris_reads_none(self, hive_store):
        assert store.read_coverage_bitmap(hive_store, convention.leaf_path(DEBRIS)) is None

    def test_missing_sidecar_reads_none(self, hive_store):
        import pathlib

        leaf = convention.leaf_path(STAMPED)
        (pathlib.Path(hive_store) / leaf / convention.COVERAGE_SIDECAR).unlink()
        assert store.read_coverage_bitmap(hive_store, leaf) is None


class TestBitmapAnd:
    def test_bitmap_hit_and_miss(self, hive_store):
        leaf = convention.leaf_path(STAMPED)
        hit = store.bitmap_and(hive_store, leaf, _words(STAMPED))  # parent covers the cell
        np.testing.assert_array_equal(hit, _words(STAMPED + "11"))
        miss = store.bitmap_and(hive_store, leaf, _words(STAMPED + "44"))  # unoccupied corner
        assert miss.size == 0  # definitive: the bitmap is exact

    def test_full_short_circuits(self, hive_store):
        leaf = convention.leaf_path(FULL, window="2019")
        hit = store.bitmap_and(hive_store, leaf, _words(FULL + "23"))
        assert hit.size == 1  # any subtree cell intersects a full shard
        miss = store.bitmap_and(hive_store, leaf, _words(STAMPED + "11"))
        assert miss.size == 0

    def test_debris_reads_none(self, hive_store):
        assert store.bitmap_and(hive_store, convention.leaf_path(DEBRIS), _words(DEBRIS)) is None


class TestWalk:
    def test_finds_exactly_the_leaves(self, hive_store):
        import pathlib

        # A non-conforming directory below the root must be ignored (node
        # invariant: not ours to interpret), as are root-only objects.
        (pathlib.Path(hive_store) / "scratch").mkdir()
        (pathlib.Path(hive_store) / "scratch" / "x.txt").write_text("x")
        found = sorted(store.walk_leaves(hive_store))
        expected = sorted(
            [
                convention.leaf_path(STAMPED),
                convention.leaf_path(FULL, window="2019"),
                convention.leaf_path(DEBRIS),
            ]
        )
        assert found == expected

    def test_paths_satisfy_node_invariant(self, hive_store):
        for rel in store.walk_leaves(hive_store):
            convention.check_node_invariant(rel)


class TestWarnIfStale:
    def test_listed_shard_not_stale(self, hive_store):
        envelope = store.load_root_coverage(hive_store)
        assert store.warn_if_stale(hive_store, STAMPED, envelope) is False

    def test_unlisted_shard_warns_once(self, hive_store):
        envelope = store.load_root_coverage(hive_store)
        with pytest.warns(UserWarning, match="root MOC lags the leaves"):
            assert store.warn_if_stale(hive_store, FULL, envelope) is True
        # Once per store per process: the second detection stays silent.
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            assert store.warn_if_stale(hive_store, FULL, envelope) is True

    def test_no_envelope_is_absence_not_staleness(self, hive_store):
        assert store.warn_if_stale(hive_store, FULL, None) is False

    def test_malformed_envelope_counts_stale(self, hive_store, recwarn):
        assert store.warn_if_stale(hive_store + "/x", FULL, {"spec": "junk"}) is True

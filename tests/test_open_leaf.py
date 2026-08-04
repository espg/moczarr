"""``open_leaf``: the leaf-direct twin of ``open_hive``.

Pins the three layers it owns — manifest-grammar path arithmetic, the
``open_object_store`` transport (same credential/anonymous policy as
``open_hive``), and the read-only ``zarr.storage.ObjectStore`` wrap — plus the
ambient-credential fallback in ``open_object_store`` itself (obstore's native
env→IMDS chain cannot see AWS profiles/SSO; the boto3 resolver can).
"""

import pytest
import zarr
from conftest import FULL, build_many_leaf_store

from moczarr import convention, open_leaf
from moczarr.store import open_object_store, read_manifest

SHARD = "-5112333"


class TestOpenLeaf:
    def test_opens_leaf_arrays(self, tmp_path):
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        store = open_leaf(root, SHARD)
        arr = zarr.open_array(store, path="8/count", mode="r")
        assert arr[:].sum() == 16  # 4**(8-6) cells of ones

    def test_packed_word_and_decimal_equivalent(self, tmp_path):
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        by_decimal = zarr.open_array(open_leaf(root, SHARD), path="8/morton", mode="r")[:]
        by_word = zarr.open_array(
            open_leaf(root, convention.morton_word(SHARD)), path="8/morton", mode="r"
        )[:]
        assert (by_decimal == by_word).all()

    def test_windowed_leaf(self, hive_store):
        group = zarr.open_group(open_leaf(hive_store, FULL, window="2019"), mode="r")
        assert convention.COMMIT_ATTR in group.attrs

    def test_manifest_threading_skips_the_get(self, tmp_path, monkeypatch):
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        manifest = read_manifest(root)

        import moczarr.open as mo

        def _boom(*a, **k):
            raise AssertionError("manifest GET should have been skipped")

        monkeypatch.setattr(mo, "read_manifest", _boom)
        store = open_leaf(root, SHARD, manifest=manifest)
        assert zarr.open_array(store, path="8/count", mode="r").shape == (16,)

    def test_missing_manifest_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="not a hive store root"):
            open_leaf(str(tmp_path / "empty"), SHARD)

    def test_read_only(self, tmp_path):
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        assert open_leaf(root, SHARD).read_only


class TestAmbientCredentialFallback:
    """open_object_store's s3 branch: boto3 resolver when nothing explicit."""

    @pytest.fixture()
    def captured(self, monkeypatch):
        calls = {}

        class _FakeS3Store:
            @staticmethod
            def from_url(url, **kwargs):
                calls.update(kwargs, _url=url)
                return object()

        monkeypatch.setattr("obstore.store.S3Store", _FakeS3Store)
        monkeypatch.setattr(
            "obstore.auth.boto3.Boto3CredentialProvider", lambda session: ("provider", session)
        )

        class _FakeSession:
            region_name = "eu-west-1"

            def get_credentials(self):
                return object()

        monkeypatch.setattr("boto3.Session", _FakeSession)
        return calls

    def test_ambient_gets_boto3_provider_and_region(self, captured):
        open_object_store("s3://bucket/prefix")
        assert captured["credential_provider"][0] == "provider"
        assert captured["region"] == "eu-west-1"

    def test_explicit_region_wins(self, captured):
        open_object_store("s3://bucket/prefix", region="us-west-2")
        assert captured["region"] == "us-west-2"

    def test_explicit_keys_bypass_the_fallback(self, captured):
        open_object_store("s3://bucket/prefix", access_key_id="AK", secret_access_key="SK")
        assert "credential_provider" not in captured
        assert "region" not in captured

    def test_anonymous_bypasses_the_fallback(self, captured):
        open_object_store("s3://bucket/prefix", anonymous=True)
        assert captured["skip_signature"] is True
        assert "credential_provider" not in captured

    def test_no_boto3_credentials_keeps_bare_store(self, captured, monkeypatch):
        class _EmptySession:
            region_name = None

            def get_credentials(self):
                return None

        monkeypatch.setattr("boto3.Session", _EmptySession)
        open_object_store("s3://bucket/prefix")
        assert "credential_provider" not in captured
        assert "region" not in captured

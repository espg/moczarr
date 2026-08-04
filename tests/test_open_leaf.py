"""``open_leaf``: the leaf-direct twin of ``open_hive``.

Pins the three layers it owns — manifest-grammar path arithmetic, the
``open_object_store`` transport (same credential/anonymous policy as
``open_hive``), and the read-only ``zarr.storage.ObjectStore`` wrap — plus the
ambient-credential fallback in ``open_object_store`` itself (obstore's native
env→IMDS chain cannot see AWS profiles/SSO; the boto3 resolver can).

The credential tests inject FAKE ``boto3`` / ``obstore.auth.boto3`` modules:
neither is a moczarr dependency, and the CI legs install neither, so the
fallback's behavior has to be pinned without them.
"""

import sys
import types
from datetime import timedelta
from pathlib import Path

import pytest
import zarr
from conftest import build_many_leaf_store

from moczarr import convention, open_leaf
from moczarr.store import _ambient_provider, open_object_store, read_manifest

SHARD = "-5112333"
#: Multi-product golden (test_products.py): ``atl06`` is /1, ``atl06_windows``
#: is /2 with 2019/2020 leaves; both shard at order 3, cells at order 5.
MULTIROOT = str(Path(__file__).parent / "data" / "multiproduct_hive")
#: ``path_grouping: 3`` golden (test_path_grouping.py): order-8 shards.
PG3 = str(Path(__file__).parent / "data" / "serc_hive_pg3")


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

    def test_windowed_leaf(self):
        group = zarr.open_group(
            open_leaf(MULTIROOT, "-5111", window="2019", product="atl06_windows"), mode="r"
        )
        assert convention.COMMIT_ATTR in group.attrs

    def test_product_reroots_on_the_subtree(self):
        # D19 sugar, the same claim open_hive's product= makes: the product
        # form equals opening the product subtree path directly.
        via_product = zarr.open_array(
            open_leaf(MULTIROOT, "4111", product="atl06"), path="5/morton", mode="r"
        )[:]
        direct = zarr.open_array(
            open_leaf(f"{MULTIROOT}/atl06", "4111"), path="5/morton", mode="r"
        )[:]
        assert (via_product == direct).all()

    def test_grouped_path_store(self):
        # path_grouping: 3 (spec §6.1) — the leaf path is the manifest's
        # grammar, so a grouped store needs no caller-side arithmetic.
        arr = zarr.open_array(open_leaf(PG3, "433142241"), path="10/count", mode="r")
        assert arr.shape == (16,)  # 4**(10-8)

    def test_manifest_threading_skips_the_get(self, tmp_path, monkeypatch):
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        manifest = read_manifest(root)

        import moczarr.open as mo

        def _boom(*a, **k):
            raise AssertionError("manifest GET should have been skipped")

        monkeypatch.setattr(mo, "read_manifest", _boom)
        store = open_leaf(root, SHARD, manifest=manifest)
        assert zarr.open_array(store, path="8/count", mode="r").shape == (16,)

    def test_threaded_manifest_is_validated(self, tmp_path):
        # A threaded manifest goes through parse_manifest like a read one:
        # {} / [] are one clear error, not a KeyError deeper in.
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        with pytest.raises(ValueError, match="unknown manifest spec"):
            open_leaf(root, SHARD, manifest={})
        with pytest.raises(ValueError, match="not a mapping"):
            open_leaf(root, SHARD, manifest=[])

    def test_shard_order_mismatch_raises(self, tmp_path):
        # The natural leaf-direct mistake: a CELL id where a shard id belongs.
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        with pytest.raises(ValueError, match=r"order-8 id.*shards at order 6"):
            open_leaf(root, SHARD + "11")

    def test_missing_manifest_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="not a hive store root"):
            open_leaf(str(tmp_path / "empty"), SHARD)

    def test_multiproduct_root_names_its_products(self):
        # Error parity with open_hive: a manifest-less MULTI-PRODUCT root is
        # not "not a hive store root", it is a root that needs product=.
        with pytest.raises(ValueError, match=r"multi-product store root.*atl06"):
            open_leaf(MULTIROOT, "4111")

    def test_window_on_unwindowed_store_raises(self, hive_store):
        with pytest.raises(ValueError, match="unwindowed stores have no window leaves"):
            open_leaf(hive_store, SHARD, window="2019")

    def test_windowed_store_without_window_raises(self):
        with pytest.raises(ValueError, match=r"is a windowed \(morton-hive/2\) store"):
            open_leaf(MULTIROOT, "-5111", product="atl06_windows")

    def test_reserved_all_token_refused(self):
        # espg/moczarr#30: every window= seam refuses the all-time token.
        with pytest.raises(ValueError, match="reserved all-time token"):
            open_leaf(MULTIROOT, "-5111", window="all", product="atl06_windows")

    def test_anonymous_reaches_the_manifest_read(self, tmp_path):
        # anonymous rides store_kwargs, so it must survive into the store the
        # manifest read builds as well as the leaf's (a local store ignores it).
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        store = open_leaf(root, SHARD, anonymous=True)
        assert zarr.open_array(store, path="8/count", mode="r").shape == (16,)

    def test_shared_handle_serves_the_manifest_read(self, tmp_path):
        # store= threads ONE root-rooted handle into the manifest GET (issue
        # #5); the leaf store itself stays a fresh, leaf-rooted open.
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        handle = open_object_store(root)
        store = open_leaf(root, SHARD, store=handle)
        assert zarr.open_array(store, path="8/count", mode="r").shape == (16,)

    def test_local_path_never_probes_boto3(self, tmp_path, monkeypatch):
        import moczarr.store as ms

        def _boom():
            raise AssertionError("a local open must never probe ambient credentials")

        monkeypatch.setattr(ms, "_ambient_provider", _boom)
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        assert zarr.open_array(open_leaf(root, SHARD), path="8/count", mode="r").shape == (16,)

    def test_read_only(self, tmp_path):
        root = build_many_leaf_store(tmp_path / "store", [SHARD])
        assert open_leaf(root, SHARD).read_only


class _Session:
    """A boto3-shaped session; credentials/region per subclass."""

    region_name: str | None = "eu-west-1"
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def get_credentials(self):
        return object()


class _NoCredentials(_Session):
    region_name = None

    def get_credentials(self):
        return None


class _StaleProfile(_Session):
    def get_credentials(self):
        raise RuntimeError("ProfileNotFound: The config profile (gone) could not be found")


class TestAmbientCredentialFallback:
    """open_object_store's s3 branch: boto3 resolver when nothing explicit."""

    @pytest.fixture(autouse=True)
    def _isolate_probe(self):
        # The probe is process-memoized (a fresh session re-mints SSO creds),
        # so each test starts and leaves it cold.
        _ambient_provider.cache_clear()
        yield
        _ambient_provider.cache_clear()

    @pytest.fixture()
    def captured(self, monkeypatch):
        calls: dict = {}

        class _FakeS3Store:
            @staticmethod
            def from_url(url, **kwargs):
                calls.clear()
                calls.update(kwargs, _url=url)
                return object()

        monkeypatch.setattr("obstore.store.S3Store", _FakeS3Store)
        self._session(monkeypatch, _Session)
        return calls

    @staticmethod
    def _session(monkeypatch, session_cls):
        """Inject fake ``boto3``/``obstore.auth.boto3`` — neither is a dep."""
        session_cls.constructions = 0
        monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(Session=session_cls))
        monkeypatch.setitem(
            sys.modules,
            "obstore.auth.boto3",
            types.SimpleNamespace(
                Boto3CredentialProvider=lambda session, **kw: ("provider", session, kw)
            ),
        )

    def test_ambient_gets_the_boto3_provider(self, captured):
        open_object_store("s3://bucket/prefix")
        assert captured["credential_provider"][0] == "provider"
        # No region kwarg: Boto3CredentialProvider carries the session's
        # region into the store config itself, and adopting it here would be
        # wrong against a custom endpoint.
        assert "region" not in captured

    def test_provider_ttl_is_short(self, captured):
        open_object_store("s3://bucket/prefix")
        assert captured["credential_provider"][2]["ttl"] == timedelta(minutes=5)

    def test_probe_is_memoized(self, captured):
        for _ in range(3):
            open_object_store("s3://bucket/prefix")
        assert _Session.constructions == 1

    def test_explicit_region_still_forwarded(self, captured):
        # A region is not a credential: it neither suppresses the provider nor
        # gets overwritten.
        open_object_store("s3://bucket/prefix", region="us-west-2")
        assert captured["region"] == "us-west-2"
        assert "credential_provider" in captured

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"access_key_id": "AK", "secret_access_key": "SK"},
            {"aws_access_key_id": "AK", "aws_secret_access_key": "SK"},
            {"token": "ST"},
            {"aws_session_token": "ST"},
            {"skip_signature": True},
            {"config": {"aws_access_key_id": "AK"}},
            {"config": {"access_key_id": "AK"}},
        ],
    )
    def test_explicit_credentials_bypass_the_fallback(self, captured, kwargs):
        # obstore accepts several spellings per option (and a config= dict);
        # each of them means the caller settled auth, so an injected provider
        # must never win over the caller's keys.
        open_object_store("s3://bucket/prefix", **kwargs)
        assert "credential_provider" not in captured

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"endpoint": "http://localhost:9000"},
            {"aws_endpoint": "http://localhost:9000"},
            {"endpoint_url": "https://x.r2.cloudflarestorage.com"},
            {"config": {"aws_endpoint_url": "https://x.r2.cloudflarestorage.com"}},
        ],
    )
    def test_custom_endpoint_bypasses_the_fallback(self, captured, kwargs):
        # MinIO/R2 want THEIR credentials and region, never the ambient AWS
        # session's — an endpoint override skips the whole injection.
        open_object_store("s3://bucket/prefix", **kwargs)
        assert "credential_provider" not in captured
        assert "region" not in captured

    def test_anonymous_bypasses_the_fallback(self, captured):
        open_object_store("s3://bucket/prefix", anonymous=True)
        assert captured["skip_signature"] is True
        assert "credential_provider" not in captured

    def test_no_boto3_credentials_keeps_bare_store(self, captured, monkeypatch):
        self._session(monkeypatch, _NoCredentials)
        open_object_store("s3://bucket/prefix")
        assert "credential_provider" not in captured
        assert "region" not in captured

    def test_no_boto3_at_all_keeps_bare_store(self, captured, monkeypatch):
        monkeypatch.setitem(sys.modules, "boto3", None)  # import raises
        open_object_store("s3://bucket/prefix")
        assert "credential_provider" not in captured

    def test_stale_profile_degrades_instead_of_raising(self, captured, monkeypatch, caplog):
        # botocore raises ProfileNotFound/ConfigParseError out of
        # get_credentials(); a stale AWS_PROFILE beside valid env keys must
        # not take down an open obstore's own chain would have served.
        self._session(monkeypatch, _StaleProfile)
        with caplog.at_level("DEBUG", logger="moczarr.store"):
            open_object_store("s3://bucket/prefix")
        assert "credential_provider" not in captured
        assert "ambient boto3 credentials unavailable" in caplog.text

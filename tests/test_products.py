"""Multi-product roots (D19 / spec §6.5): discovery + the ``open_hive`` sugar.

``tests/data/multiproduct_hive`` is the committed golden from
``tools/generate_multiproduct_fixture.py``: three named products under one
store root (``atl06`` — /1, ``aggregation.yaml`` + ``semantic_hash``;
``atl06_windows`` — /2 windowed, neither; ``atl06_ragged`` — /1 carrying the
vlen O11 arrays) plus a name-shaped non-product child (``scratch/``). The
bare single-product form is the existing
``serc_hive`` fixture — its whole suite is the "single-product unchanged"
guard; the pins here are the two root forms' discrimination by content.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moczarr import list_products, open_hive
from moczarr.products import is_product_name, validate_product_name

FIXTURE = Path(__file__).parent / "data" / "multiproduct_hive"
SERC = Path(__file__).parent / "data" / "serc_hive"


@pytest.fixture()
def multiroot():
    return str(FIXTURE)


class TestProductNameGrammar:
    def test_accepts_spec_charset(self):
        for name in ("atl06", "atl06_windows", "a", "x" * 192, "a-b_c9"):
            assert is_product_name(name)
            assert validate_product_name(name) == name

    def test_rejects_base_components(self):
        # The §6.5 base-component exclusion: the walker's child
        # classification must stay unambiguous.
        for name in ("1", "6", "-1", "-6", "4"):
            assert not is_product_name(name)

    def test_rejects_off_grammar(self):
        for name in ("", "x" * 193, "UPPER", "sp ace", "dot.ted", "a/b", "café"):
            assert not is_product_name(name)
            with pytest.raises(ValueError, match="§6.5"):
                validate_product_name(name)

    def test_digit_names_beyond_base_grammar_are_legal(self):
        # Only -?[1-6] is excluded; other digit-y names are fine per spec.
        for name in ("7", "0", "12", "-12", "2019"):
            assert is_product_name(name)


class TestListProducts:
    def test_enumerates_named_products(self, multiroot):
        products = list_products(multiroot)
        names = [p["name"] for p in products]
        assert names == ["atl06", "atl06_ragged", "atl06_windows"]  # sorted

    def test_surfaces_semantic_hash_and_core_presence(self, multiroot):
        by_name = {p["name"]: p for p in list_products(multiroot)}
        atl06 = by_name["atl06"]
        assert atl06["spec"] == "morton-hive/1"
        assert atl06["has_aggregation_yaml"] is True
        # D19 self-consistency: the manifest hash IS the sha256 of the
        # derived aggregation.yaml bytes (the hash is truth; the YAML a
        # regenerable convenience).
        import hashlib

        core = (FIXTURE / "atl06" / "aggregation.yaml").read_bytes()
        assert atl06["semantic_hash"] == hashlib.sha256(core).hexdigest()
        windows = by_name["atl06_windows"]
        assert windows["spec"] == "morton-hive/2"
        assert windows["semantic_hash"] is None
        assert windows["has_aggregation_yaml"] is False
        assert windows["manifest"]["temporal"] == {"schedule": "yearly"}

    def test_skips_name_shaped_child_without_manifest(self, multiroot):
        # scratch/ satisfies the name grammar but carries no manifest.
        assert "scratch" not in {p["name"] for p in list_products(multiroot)}

    def test_bare_store_has_no_named_products(self):
        # §6.5 content discrimination: a manifest at the root ⇒ bare store.
        assert list_products(str(SERC)) == []

    def test_malformed_product_manifest_raises(self, multiroot, tmp_path):
        # A present-but-garbage manifest is a broken bootstrap, never a
        # silently skipped product (read_manifest's loud posture).
        root = tmp_path / "store"
        shutil.copytree(FIXTURE, root)
        (root / "atl06" / "morton_hive.json").write_text('{"spec": "nope/9"}')
        with pytest.raises(ValueError, match="unknown manifest spec"):
            list_products(str(root))


class TestOpenHiveProduct:
    def test_product_equals_subtree_open(self, multiroot):
        # The D19 sugar is EXACTLY the subtree open (a product subtree is a
        # complete, unmodified morton-hive store).
        via_product = open_hive(multiroot, product="atl06")
        via_path = open_hive(f"{multiroot}/atl06")
        xr.testing.assert_identical(via_product, via_path)
        assert via_product.sizes["cells"] == 32  # 2 shards x 16 cells

    def test_products_are_independent_stores(self, multiroot):
        counts = open_hive(multiroot, product="atl06")
        heights = open_hive(multiroot, product="atl06_windows", window="2019")
        assert set(counts.data_vars) == {"count"}
        assert set(heights.data_vars) == {"height"}
        assert heights.sizes["cells"] == 32
        assert np.asarray(heights["height"].values, dtype=np.float64).max() > 0

    def test_multiproduct_root_without_product_is_pointed(self, multiroot):
        with pytest.raises(ValueError, match=r"multi-product store root.*atl06.*atl06_windows"):
            open_hive(multiroot)

    def test_non_hive_root_error_unchanged(self, tmp_path):
        # A directory with neither manifest nor products keeps the plain
        # "not a hive store root" error.
        (tmp_path / "junk.txt").write_text("nope")
        with pytest.raises(ValueError, match="not a hive store root"):
            open_hive(str(tmp_path))

    def test_missing_product_errors(self, multiroot):
        # Local stores fail at the store-open (no such directory); the
        # message names the product path either way.
        with pytest.raises((ValueError, FileNotFoundError), match="atl07"):
            open_hive(multiroot, product="atl07")

    def test_off_grammar_product_rejected_before_io(self, multiroot):
        with pytest.raises(ValueError, match="§6.5"):
            open_hive(multiroot, product="../escape")

    def test_product_on_bare_store_is_not_a_store(self):
        # serc_hive/{name} does not exist; the subtree open fails plainly.
        with pytest.raises((ValueError, FileNotFoundError)):
            open_hive(str(SERC), product="atl06")

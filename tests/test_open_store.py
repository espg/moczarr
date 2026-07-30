"""``open_store``: the DataTree-shaped opener (issue #15, phase 8a).

Runs against the PR-B two-product fixture (``tests/data/multiproduct_hive``:
``atl06`` — /1, ``count``, ``semantic_hash``; ``atl06_windows`` — /2
windowed, ``height``, no hash) and the bare SERC fixture for the one-child
degenerate form. The pins: one child per product holding exactly the
per-node ``open_hive`` Dataset, an empty root, heterogeneous sibling
schemas, per-node AOI scoping, and laziness (zero chunk GETs at open, the
``test_open.py`` zero-read style).
"""

import json
import warnings
from pathlib import Path

import pytest
import xarray as xr

from moczarr import open_hive, open_store

FIXTURE = Path(__file__).parent / "data" / "multiproduct_hive"
SERC = Path(__file__).parent / "data" / "serc_hive"


@pytest.fixture()
def multiroot():
    return str(FIXTURE)


def _semantic_hash(product_root: Path) -> str:
    return json.loads((product_root / "morton_hive.json").read_text())["semantic_hash"]


class TestOpenStoreTree:
    def test_two_product_tree(self, multiroot):
        tree = open_store(multiroot, window="2019")
        assert isinstance(tree, xr.DataTree)
        assert list(tree.children) == ["atl06", "atl06_windows"]  # name-sorted

    def test_root_is_empty_with_store_attrs(self, multiroot):
        # Root node: no data, no coords — store-level attrs only (the
        # multi-product root is manifest-less by design; O14).
        tree = open_store(multiroot, window="2019")
        root = tree.to_dataset()
        assert not root.data_vars
        assert not root.coords
        assert tree.attrs["morton_hive_store"] == {
            "store_root": multiroot,
            "products": ["atl06", "atl06_windows"],
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="defaults"),
            pytest.param({"decode": True}, id="decode"),
            pytest.param({"index_kind": "pandas", "aoi": ["4111"]}, id="pandas-aoi"),
            pytest.param({"concurrency": 1, "fabricate_cell_ids": False}, id="serial-nofabricate"),
            pytest.param(
                {"xr_kwargs": {"mask_and_scale": False}, "anonymous": False},
                id="xr_kwargs-anonymous",
            ),
        ],
    )
    def test_nodes_are_the_per_product_open_hive_datasets(self, multiroot, kwargs):
        # Each child holds EXACTLY what open_hive returns for that product
        # (plus the semantic_hash attr the tree layer surfaces) — pinned
        # across kwarg bundles so the forwarding block cannot rot silently
        # (every kwarg open_store forwards is exercised by some bundle).
        if kwargs.get("decode"):
            pytest.importorskip("moczarr.dggs")
        with warnings.catch_warnings():
            # the aoi bundle empties atl06_windows (issue #4, per node)
            warnings.simplefilter("ignore", UserWarning)
            tree = open_store(multiroot, window="2019", **kwargs)
            want_plain = open_hive(multiroot, product="atl06", **kwargs)
            want_windowed = open_hive(multiroot, product="atl06_windows", window="2019", **kwargs)
        want_plain.attrs["semantic_hash"] = _semantic_hash(FIXTURE / "atl06")
        xr.testing.assert_identical(tree["atl06"].to_dataset(), want_plain)
        xr.testing.assert_identical(tree["atl06_windows"].to_dataset(), want_windowed)

    def test_schema_heterogeneity_across_nodes(self, multiroot):
        # Sibling nodes carry different variable sets — the non-alignable-
        # groups case a flat Dataset cannot represent.
        tree = open_store(multiroot, window="2019")
        assert set(tree["atl06"].ds.data_vars) == {"count"}
        assert set(tree["atl06_windows"].ds.data_vars) == {"height"}

    def test_attrs_carry_semantic_hash(self, multiroot):
        tree = open_store(multiroot, window="2019")
        assert tree["atl06"].ds.attrs["semantic_hash"] == _semantic_hash(FIXTURE / "atl06")
        # No hash in the manifest -> no attr fabricated (never None-valued).
        assert "semantic_hash" not in tree["atl06_windows"].ds.attrs
        # The per-node manifest summary rides along unchanged.
        assert tree["atl06"].ds.attrs["morton_hive"]["spec"] == "morton-hive/1"
        assert tree["atl06_windows"].ds.attrs["morton_hive"]["spec"] == "morton-hive/2"

    def test_per_node_aoi(self, multiroot):
        # One AOI, scoped per node by each product's own coverage: it covers
        # an atl06 shard (4111) and none of atl06_windows (-511*), so the
        # windowed node is the issue-#4 schema-correct empty — per node.
        with pytest.warns(UserWarning, match="intersects no coverage"):
            tree = open_store(multiroot, window="2019", aoi=["4111"])
        assert tree["atl06"].ds.sizes["cells"] == 16
        assert tree["atl06_windows"].ds.sizes["cells"] == 0
        assert set(tree["atl06_windows"].ds.data_vars) == {"height"}


class TestOpenStoreProductsFilter:
    def test_filter_selects_children(self, multiroot):
        # Filtered to the unwindowed product, no window= is needed at all.
        tree = open_store(multiroot, products=["atl06"])
        assert list(tree.children) == ["atl06"]
        assert tree.attrs["morton_hive_store"]["products"] == ["atl06", "atl06_windows"]

    def test_unknown_product_raises_with_roster(self, multiroot):
        with pytest.raises(ValueError, match=r"atl07.*atl06.*atl06_windows"):
            open_store(multiroot, products=["atl07"])

    def test_off_grammar_product_rejected_before_io(self, multiroot):
        # "Before I/O" is literal: the §6.5 grammar check runs ahead of the
        # object-store open, so no LIST/GET is spent on a caller typo.
        import moczarr.open as open_mod

        with pytest.raises(ValueError, match="§6.5"):
            open_store(multiroot, products=["../escape"])

        def boom(*args, **kwargs):
            raise AssertionError("store opened before the product grammar check")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(open_mod, "open_object_store", boom)
            with pytest.raises(ValueError, match="§6.5"):
                open_store(multiroot, products=["../escape"])


class TestOpenStoreWindowForwarding:
    def test_window_reaches_only_windowed_products(self, multiroot):
        # The tree-level refinement: /1 nodes hold the whole product, /2
        # nodes are window-scoped — one call opens the mixed store.
        tree = open_store(multiroot, window="2020")
        assert tree["atl06"].ds.sizes["cells"] == 32  # whole product
        assert tree["atl06_windows"].ds.sizes["cells"] == 16  # 2020 has one leaf

    def test_windowed_product_without_window_raises_labels(self, multiroot):
        # open_hive's per-product posture propagates: the windowed node
        # demands window= and names the labels that exist.
        with pytest.raises(ValueError, match=r"window=.*2019.*2020"):
            open_store(multiroot)

    def test_window_with_no_windowed_product_raises(self, multiroot):
        # Typo-catching preserved: a window against a selection with no
        # windowed product is an error, not a silent no-op.
        with pytest.raises(ValueError, match="no windowed"):
            open_store(multiroot, products=["atl06"], window="2019")


class TestOpenStoreDegenerate:
    def test_bare_store_is_a_one_child_tree(self):
        # The valid degenerate form, pinned: a bare single-product store
        # opens as a one-child tree, the node named from the manifest's
        # dataset short_name ("ATL06" -> "atl06").
        tree = open_store(str(SERC))
        assert list(tree.children) == ["atl06"]
        assert not tree.to_dataset().data_vars
        want = open_hive(str(SERC))
        want.attrs["semantic_hash"] = _semantic_hash(SERC)
        xr.testing.assert_identical(tree["atl06"].to_dataset(), want)

    def test_bare_store_roster_agrees_with_list_products(self):
        # The root attrs never claim a roster list_products denies: a bare
        # store publishes no named products, so products == [] and the
        # derived child identity is reported as `node` under `bare: True`.
        from moczarr import list_products

        tree = open_store(str(SERC))
        assert list_products(str(SERC)) == []
        assert tree.attrs["morton_hive_store"] == {
            "store_root": str(SERC),
            "products": [],
            "bare": True,
            "node": "atl06",
        }

    def test_products_filter_against_bare_store_raises(self):
        # No roster to filter on — including the derived node name, which is
        # not a product (the old behavior "succeeded" against it).
        for names in (["atl06"], ["serc"]):
            with pytest.raises(ValueError, match="bare single-product store"):
                open_store(str(SERC), products=names)

    def test_not_a_store_raises(self, tmp_path):
        (tmp_path / "junk.txt").write_text("nope")
        with pytest.raises(ValueError, match="not a hive store root"):
            open_store(str(tmp_path))


class TestOpenStoreKwargsForwarding:
    def test_index_kind_forwards_per_node(self, multiroot):
        from moczarr.moc_index import MortonMocIndex

        tree = open_store(multiroot, window="2019")  # moc default
        assert isinstance(tree["atl06"].ds.xindexes["morton"], MortonMocIndex)
        assert isinstance(tree["atl06_windows"].ds.xindexes["morton"], MortonMocIndex)
        tree = open_store(multiroot, window="2019", index_kind="pandas")
        assert "morton" not in tree["atl06"].ds.xindexes  # the index-free posture

    def test_fabricate_cell_ids_forwards(self, multiroot):
        tree = open_store(multiroot, window="2019", fabricate_cell_ids=False)
        for name in ("atl06", "atl06_windows"):
            assert "cell_ids" not in tree[name].ds.variables

    def test_laziness_preserved_per_node(self, multiroot, monkeypatch):
        # Laziness is exactly open_hive's, per node: an open_store GETs no
        # coordinate chunks (the moc-default zero-read pin, test_open.py
        # style), and the tree layer itself adds NO chunk reads at all —
        # the chunk GETs of the whole open_store equal those of the
        # per-product open_hive calls it wraps.
        import obstore

        keys: list[str] = []
        real_get, real_get_async = obstore.get, obstore.get_async

        def record(store_, key, *args, **kwargs):
            keys.append(key)
            return real_get(store_, key, *args, **kwargs)

        def record_async(store_, key, *args, **kwargs):
            keys.append(key)
            return real_get_async(store_, key, *args, **kwargs)

        monkeypatch.setattr(obstore, "get", record)
        monkeypatch.setattr(obstore, "get_async", record_async)

        def chunk_reads():
            # Chunk objects live under a "c" path component; digit-tree
            # components are 1..4 and metadata objects are *.json/.moc,
            # so "/c/" appears in chunk keys only.
            return sorted(k for k in keys if "/c/" in k)

        open_store(multiroot, window="2019")
        via_store = chunk_reads()
        assert [k for k in via_store if "/morton/c" in k or "/cell_ids/c" in k] == []
        keys.clear()
        open_hive(multiroot, product="atl06")
        open_hive(multiroot, product="atl06_windows", window="2019")
        assert via_store == chunk_reads()

    def test_single_leaf_nodes_stay_fully_lazy(self, multiroot, monkeypatch):
        # AOI-scoped to one leaf per product, nothing is concatenated and
        # no chunk object is fetched at open; the data read flows on demand.
        import obstore

        keys: list[str] = []
        real_get, real_get_async = obstore.get, obstore.get_async

        def record(store_, key, *args, **kwargs):
            keys.append(key)
            return real_get(store_, key, *args, **kwargs)

        def record_async(store_, key, *args, **kwargs):
            keys.append(key)
            return real_get_async(store_, key, *args, **kwargs)

        monkeypatch.setattr(obstore, "get", record)
        monkeypatch.setattr(obstore, "get_async", record_async)
        tree = open_store(multiroot, products=["atl06"], aoi=["4111"])
        assert [k for k in keys if "/c/" in k] == []
        assert int(tree["atl06"].ds["count"].sum()) > 0  # the read still flows
        assert [k for k in keys if "/c/" in k] != []

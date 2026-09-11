"""The multi-order assembly: a DataTree over ``pyramid_levels`` (issue #36b).

Runs on BOTH grammars, per the standing note that the ``/1`` live stores are
first-class: the swept ``tests/data/overview_hive`` fixture
(``tools/generate_overview_fixture.py``, zagg-written end to end) for the
``/1`` constant-depth ladder, and the vendored ``/2`` spec fixtures — plus
``test_level.py``'s hand-built stage artifact and sibling column (spec
grammar, not writer bytes; the deferral is recorded there) — for the fixed
ladder with its column tier. The assembly law under test: the tree's groups
are exactly the materialized resolution levels, each group is byte-identical
to the :func:`moczarr.open_level` Dataset for that resolution, and the
declaration record rides the root.
"""

import shutil
import warnings
from pathlib import Path

import pytest
import xarray as xr
from test_level import _write_sibling_column, _write_stage_artifact

from moczarr import LEVEL_ATTR, open_level, open_pyramid, read_manifest

SPEC = Path(__file__).parent / "data" / "spec"
TEMPORAL = str(SPEC / "temporal")
OVERVIEW = Path(__file__).parent / "data" / "overview_hive"
SERC = Path(__file__).parent / "data" / "serc_hive"


@pytest.fixture()
def swept(tmp_path):
    """The /2 fixture with one swept rung (cells 4) and a second leaf column."""
    root = tmp_path / "hive"
    shutil.copytree(TEMPORAL, root)
    _write_stage_artifact(root)
    _write_sibling_column(root)
    return str(root)


def _quiet_tree(root, **kwargs):
    """Assemble, swallowing the unswept-rung warnings the fixture implies."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return open_pyramid(root, **kwargs)


class TestAssemblyV2:
    def test_groups_are_the_materialized_levels(self, swept):
        # Declared levels: 6 (source), 5 (column), 4..1 (rungs). Materialized:
        # 6, 5, and the one swept rung at 4 — the unswept rungs are omitted,
        # finest first, named by the integer cell order they store.
        tree = _quiet_tree(swept)
        assert isinstance(tree, xr.DataTree)
        assert list(tree.children) == ["6", "5", "4"]

    def test_each_group_is_the_open_level_dataset(self, swept):
        tree = _quiet_tree(swept)
        for r in (6, 5, 4):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                want = open_level(swept, r)
            xr.testing.assert_identical(tree[str(r)].to_dataset(), want)
            assert tree[str(r)].ds.attrs[LEVEL_ATTR]["cell_order"] == r

    def test_root_is_empty_with_the_declaration(self, swept):
        tree = _quiet_tree(swept)
        root = tree.to_dataset()
        assert not root.data_vars and not root.coords
        manifest = read_manifest(swept)
        assert tree.attrs["morton_hive"] == {
            "spec": "morton-hive/1",
            "cell_order": 6,
            "shard_order": 4,
            "dataset": manifest["dataset"],
        }
        assert tree.attrs["semantic_hash"] == manifest["semantic_hash"]
        decl = tree.attrs["zagg_pyramid"]
        assert decl["spec"] == "zagg-pyramid/2"
        assert decl["leaf_cells"] == [5]
        # The mirror is not fabricated: this manifest records none.
        assert "multiscales" not in tree.attrs

    def test_unmaterialized_levels_warn_and_are_omitted(self, swept):
        with pytest.warns(UserWarning, match="no stamped overview object"):
            tree = open_pyramid(swept)
        assert "3" not in tree.children and "1" not in tree.children

    def test_levels_selector_bounds_the_assembly(self, swept):
        tree = _quiet_tree(swept, levels=[6, 5])
        assert list(tree.children) == ["6", "5"]

    def test_levels_selector_refuses_a_non_level(self, swept):
        with pytest.raises(ValueError, match=r"levels are \[6, 5, 4, 3, 2, 1\]"):
            open_pyramid(swept, levels=[7])

    def test_empty_levels_selector_raises(self, swept):
        with pytest.raises(ValueError, match=r"levels=\[\] selects nothing"):
            open_pyramid(swept, levels=[])

    def test_aoi_scopes_rows_never_the_tree_shape(self, swept):
        # The AOI covers only the northern leaf: the southern sibling's rows
        # are cut from the column level, but the groups stay the groups.
        tree = _quiet_tree(swept, aoi=["11213"])
        assert list(tree.children) == ["6", "5", "4"]
        assert tree["5"].ds.sizes["cells"] == 4  # the one covered leaf's cells
        full = _quiet_tree(swept)
        assert full["5"].ds.sizes["cells"] == 8  # both leaves

    def test_multiscales_mirror_rides_verbatim(self, swept):
        # The §4.9 discovery mirror is surfaced AS RECORDED when the manifest
        # carries one — never consulted (levels come from the pyramid block),
        # never re-derived.
        manifest = read_manifest(swept)
        mirror = [{"spec": "zagg-multiscales/1", "name": "SPEC_FIXTURE"}]
        manifest["multiscales"] = mirror
        tree = _quiet_tree(swept, manifest=manifest, levels=[6])
        assert tree.attrs["multiscales"] == mirror


class TestAssemblyV1:
    ATL06 = str(OVERVIEW / "atl06")

    def test_constant_depth_ladder(self):
        tree = open_pyramid(self.ATL06)
        assert list(tree.children) == ["8", "6", "4"]
        for r in (8, 6, 4):
            xr.testing.assert_identical(tree[str(r)].to_dataset(), open_level(self.ATL06, r))
        assert tree.attrs["zagg_pyramid"]["spec"] == "zagg-pyramid/1"
        assert tree.attrs["morton_hive"]["cell_order"] == 8

    def test_windowed_product_scopes_by_window(self):
        root = str(OVERVIEW / "atl06_windows")
        tree = open_pyramid(root, window="2019")
        assert list(tree.children) == ["8", "6"]
        assert tree["6"].ds.attrs["morton_hive"]["cell_order"] == 6
        with pytest.raises(ValueError, match="window="):
            open_pyramid(root)

    def test_declared_off_store_is_the_one_level_tree(self):
        # The valid degenerate form: every store is at least
        # single-resolution, and a declared-off root carries no zagg_pyramid.
        tree = open_pyramid(str(SERC))
        assert list(tree.children) == ["8"]
        assert "zagg_pyramid" not in tree.attrs
        xr.testing.assert_identical(tree["8"].to_dataset(), open_level(str(SERC), 8))

    def test_multi_product_root(self):
        tree = open_pyramid(str(OVERVIEW), product="atl06")
        assert list(tree.children) == ["8", "6", "4"]
        with pytest.raises(ValueError, match="multi-product store root"):
            open_pyramid(str(OVERVIEW))

    def test_one_root_moc_read_for_the_whole_ladder(self, monkeypatch):
        # issue #5 at the tree layer: the sidecar tier must not grow with the
        # number of levels — one shared read for the non-source arms, plus
        # open_hive's own on the source arm.
        import obstore

        keys: list[str] = []
        real_get = obstore.get

        def record(store_, key, *args, **kwargs):
            keys.append(key)
            return real_get(store_, key, *args, **kwargs)

        monkeypatch.setattr(obstore, "get", record)
        open_pyramid(self.ATL06)
        moc_reads = [k for k in keys if k.endswith("coverage.moc") and "/" not in k]
        assert len(moc_reads) <= 2

    def test_tree_layer_adds_no_reads(self, monkeypatch):
        # Laziness is each level's own (the open_store posture, per level):
        # the chunk GETs of the whole assembly equal those of the per-level
        # open_level calls it wraps — the tree layer itself reads nothing —
        # and no morton/cell_ids coordinate chunk is fetched anywhere (the
        # moc-default zero-coordinate-read pin, test_open.py style).
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
            return sorted(k for k in keys if "/c/" in k)

        open_pyramid(self.ATL06)
        via_tree = chunk_reads()
        assert [k for k in via_tree if "/morton/c" in k or "/cell_ids/c" in k] == []
        keys.clear()
        for r in (8, 6, 4):
            open_level(self.ATL06, r)
        assert via_tree == chunk_reads()

    def test_manifest_and_store_thread_every_level(self):
        # The threaded pair governs the whole assembly: a bogus store_root
        # proves no level re-resolves its own store, and the threaded
        # manifest is the manifest used.
        from moczarr import open_object_store

        handle = open_object_store(self.ATL06)
        manifest = read_manifest(self.ATL06)
        manifest["dataset"] = {"short_name": "THREADED"}
        tree = open_pyramid("/nonexistent/root", store=handle, manifest=manifest)
        assert list(tree.children) == ["8", "6", "4"]
        assert tree.attrs["morton_hive"]["dataset"] == {"short_name": "THREADED"}
        for child in tree.children.values():
            assert child.ds.attrs["morton_hive"]["dataset"] == {"short_name": "THREADED"}


class TestOpenStoreV2:
    """open_store's /2 order nodes (issue #36b scope (2))."""

    def test_bare_v2_store_grows_level_children(self, swept):
        from moczarr import open_store

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tree = open_store(swept)
        assert list(tree.children) == ["spec_fixture"]
        product = tree["spec_fixture"]
        # The product node is an empty intermediate carrying the identity
        # attrs plus the normalized declaration; the children are the
        # MATERIALIZED resolution levels, finest first.
        assert not product.to_dataset().data_vars
        assert product.ds.attrs["zagg_pyramid"]["spec"] == "zagg-pyramid/2"
        assert product.ds.attrs["morton_hive"]["cell_order"] == 6
        assert "semantic_hash" in product.ds.attrs
        assert list(product.children) == ["6", "5", "4"]

    def test_children_are_the_open_level_datasets(self, swept):
        from moczarr import open_store

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tree = open_store(swept)
            for r in (6, 5, 4):
                want = open_level(swept, r)
                xr.testing.assert_identical(tree["spec_fixture"][str(r)].to_dataset(), want)

    def test_decode_refused_on_a_v2_product(self, swept):
        from moczarr import open_store

        with pytest.raises(ValueError, match="zagg#550"):
            open_store(swept, decode=True)

    def test_unknown_pyramid_revision_fails_loudly(self, swept, tmp_path):
        # A future /3 block must not read as declared-off (§4.5's conformance
        # rule) — open_store now runs the same strict check open_level does.
        import json

        from moczarr import open_store

        root = tmp_path / "hive3"
        shutil.copytree(swept, root)
        manifest_path = root / "morton_hive.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["pyramid"]["spec"] = "zagg-pyramid/3"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="zagg-pyramid/3"):
            open_store(str(root))

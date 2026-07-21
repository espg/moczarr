"""The ``path_grouping: 3`` golden store (``tests/data/serc_hive_pg3``).

Committed fixture from ``tools/generate_pg3_fixture.py`` (moczarr-built —
zagg's writer declares the D21 manifest field but cannot yet WRITE grouped
paths; see the tool docstring). These goldens exercise the generic chunked
path end-to-end — arithmetic path construction, the discovery walk's child
classification, and ``open_hive`` — including the short remainder component
(shard order 8, grouping 3 → components ``3+3+2``). The grouping==1
behavior guard is the entire existing suite over ``serc_hive``.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from moczarr import convention, open_hive, store

FIXTURE = Path(__file__).parent / "data" / "serc_hive_pg3"
#: The fixture's shards and their golden grouped paths (spec §6.1 chunking).
GOLDEN_PATHS = {
    "433142241": "4/331/422/41/433142241.zarr",
    "433142242": "4/331/422/42/433142242.zarr",
    "-433412214": "-4/334/122/14/-433412214.zarr",
}


@pytest.fixture()
def pg3():
    return str(FIXTURE)


def test_manifest_declares_grouping(pg3):
    manifest = store.read_manifest(pg3)
    assert manifest["path_grouping"] == 3
    assert convention.manifest_path_grouping(manifest) == 3


def test_golden_paths_exist(pg3):
    # The committed tree IS the golden vector: the arithmetic paths point at
    # real on-disk leaves, remainder component included.
    for shard, rel in GOLDEN_PATHS.items():
        assert convention.leaf_path(shard, path_grouping=3) == rel
        assert (Path(pg3) / rel / "zarr.json").is_file()


def test_walk_classifies_grouped_children(pg3):
    leaves = sorted(store.walk_leaves(pg3, path_grouping=3))
    assert leaves == sorted(GOLDEN_PATHS.values())
    # Concurrency path: same SET (order may differ, per the walk contract).
    assert sorted(store.walk_leaves(pg3, concurrency=8, path_grouping=3)) == leaves


def test_walk_default_grouping_misses_grouped_nodes(pg3):
    # Why threading the manifest value matters: a walker assuming one digit
    # per level cannot classify 3-digit children and finds nothing.
    assert list(store.walk_leaves(pg3)) == []


def test_open_hive_arithmetic_path(pg3):
    ds = open_hive(pg3)
    assert ds.sizes["cells"] == 48  # 3 shards x 16 cells
    words = np.asarray(ds["morton"].values, dtype=np.uint64)
    assert np.all(np.diff(words.astype(object)) > 0)  # ascending packed order
    # Every cell sits under its shard (prefix = ancestor).
    decimals = [convention.morton_decimal(int(w)) for w in words]
    assert {d[:-2] for d in decimals} == set(GOLDEN_PATHS)


def test_open_hive_aoi_subset(pg3):
    ds = open_hive(pg3, aoi=["433142241"])
    assert ds.sizes["cells"] == 16
    decimals = [convention.morton_decimal(int(w)) for w in np.asarray(ds["morton"].values)]
    assert {d[:-2] for d in decimals} == {"433142241"}


def test_open_hive_walk_fallback(pg3, tmp_path):
    # No root MOC: the discovery walk (with the manifest's grouping) is the
    # candidate source and the open still succeeds.
    copy = tmp_path / "pg3_no_moc"
    shutil.copytree(FIXTURE, copy)
    (copy / convention.ROOT_COVERAGE_NAME).unlink()
    ds = open_hive(str(copy))
    assert ds.sizes["cells"] == 48


def test_open_hive_empty_aoi_schema(pg3):
    # The issue #4 empty contract composes with grouped paths (_schema_leaf
    # threads the grouping too).
    with pytest.warns(UserWarning, match="intersects no coverage"):
        ds = open_hive(pg3, aoi=["1111"])
    assert ds.sizes["cells"] == 0
    assert "count" in ds.data_vars


def test_moc_index_on_grouped_store(pg3):
    ds = open_hive(pg3, index_kind="moc")
    assert ds.sizes["cells"] == 48
    dense = open_hive(pg3)
    np.testing.assert_array_equal(ds["morton"].values, dense["morton"].values)

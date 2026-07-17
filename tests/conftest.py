"""Shared fixture: a hand-built local hive store matching the mortie#62 layout.

Built object-by-object (no zagg dependency) so the tests document the wire
format explicitly; the byte-level conventions are pinned by the golden
vectors in ``test_coverage.py``.
"""

import json
from pathlib import Path

import pytest
from numcodecs import Zstd

from moczarr import convention

#: The three shards in the fixture store (order 6, southern base -5).
STAMPED = "-5112333"  # bitmap-encoding leaf, occupied cell = shard + "11"
FULL = "-5112331"  # windowed leaf (2019), encoding "full"
DEBRIS = "-5112334"  # leaf prefix without a commit stamp


def _manifest(cell_order=8, shard_order=6):
    return {
        "spec": convention.HIVE_SPEC,
        "dataset": {"short_name": "ATL06", "version": "007"},
        "cell_order": cell_order,
        "shard_order": shard_order,
        "split_schedule": [1] * shard_order,
        "pyramid": {"orders": [], "aggregation": {}},
        "generated_at": "2026-07-17T00:00:00+00:00",
    }


def _root_coverage():
    # Deliberately lists only STAMPED and DEBRIS' run sibling — FULL is left
    # unlisted so warn_if_stale has a stale case to detect.
    return {
        "spec": "morton-moc/1",
        "encoding": "ranges",
        "order": 6,
        "source": "dispatcher",
        "generated_at": "2026-07-17T00:00:00+00:00",
        "ranges": [[STAMPED, STAMPED]],
    }


def _stamp(coverage, *, window=None):
    stamp = {
        "spec": convention.HIVE_SPEC if window is None else convention.HIVE_SPEC_V2,
        "complete": True,
        "cells_with_data": 1,
        "granule_count": 1,
        "written_at": "2026-07-17T00:00:00+00:00",
        "coverage": coverage,
    }
    if window is not None:
        stamp["window"] = window
    return stamp


def _zarr_group(attrs):
    return {"zarr_format": 3, "node_type": "group", "attributes": attrs}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _array_meta(data_type, length):
    """Uncompressed zarr v3 array metadata: raw little-endian chunk bytes."""
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [length],
        "data_type": data_type,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [length]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "dimension_names": ["cells"],
    }


def build_many_leaf_store(root, shards, *, cell_order=8, shard_order=6):
    """A synthetic hive store with one real (openable) zarr leaf per shard.

    The issue #5 concurrency fixture: leaves are written object-by-object
    like :func:`hive_store` — raw v3 metadata plus uncompressed chunk bytes,
    no zarr machinery — so a few hundred leaves build in well under a
    second. Every leaf is fully occupied (``encoding: "full"``, no sidecar):
    ``morton`` holds all ``4**(cell_order - shard_order)`` subtree cells in
    ascending packed order and ``count`` is all ones. Returns ``str(root)``.
    """
    import itertools

    import numpy as np

    root = Path(root)
    depth = cell_order - shard_order
    suffixes = ["".join(p) for p in itertools.product("1234", repeat=depth)]
    _write_json(root / convention.MANIFEST_NAME, _manifest(cell_order, shard_order))
    _write_json(
        root / convention.ROOT_COVERAGE_NAME,
        {
            "spec": "morton-moc/1",
            "encoding": "ranges",
            "order": shard_order,
            "source": "dispatcher",
            "generated_at": "2026-07-17T00:00:00+00:00",
            "ranges": [[s, s] for s in shards],
        },
    )
    group_meta = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    morton_meta = json.dumps(_array_meta("uint64", len(suffixes)))
    count_meta = json.dumps(_array_meta("int64", len(suffixes)))
    count_bytes = np.ones(len(suffixes), dtype="<i8").tobytes()
    for shard in shards:
        coverage = {
            "spec": "morton-moc/1",
            "box": [shard, None, None, None],
            "cell_order": cell_order,
            "source": "worker",
            "encoding": "full",
        }
        leaf = root / convention.leaf_path(shard)
        _write_json(leaf / "zarr.json", _zarr_group({convention.COMMIT_ATTR: _stamp(coverage)}))
        words = np.sort(
            np.asarray([convention.morton_word(shard + s) for s in suffixes], dtype=np.uint64)
        )
        group_dir = leaf / str(cell_order)
        for name, meta, chunk in (
            ("morton", morton_meta, words.astype("<u8").tobytes()),
            ("count", count_meta, count_bytes),
        ):
            (group_dir / name / "c").mkdir(parents=True)
            (group_dir / name / "zarr.json").write_text(meta)
            (group_dir / name / "c" / "0").write_bytes(chunk)
        (group_dir / "zarr.json").write_text(group_meta)
    return str(root)


@pytest.fixture()
def hive_store(tmp_path):
    """A local store root with one bitmap leaf, one full windowed leaf, debris."""
    root = tmp_path / "store"
    _write_json(root / convention.MANIFEST_NAME, _manifest())
    _write_json(root / convention.ROOT_COVERAGE_NAME, _root_coverage())

    # STAMPED: bitmap-encoding leaf; occupied cell = shard + "11" (rank 0,
    # the b"\x80\x00" golden raw bitmap at depth 2).
    bitmap = bytes(Zstd(level=3).encode(b"\x80\x00"))
    leaf = root / convention.leaf_path(STAMPED)
    coverage = {
        "spec": "morton-moc/1",
        "box": [STAMPED + "11", None, None, None],
        "cell_order": 8,
        "source": "worker",
        "encoding": "bitmap",
        "sidecar": convention.COVERAGE_SIDECAR,
        "nbytes": len(bitmap),
        "raw_nbytes": 2,
    }
    _write_json(leaf / "zarr.json", _zarr_group({convention.COMMIT_ATTR: _stamp(coverage)}))
    (leaf / convention.COVERAGE_SIDECAR).write_bytes(bitmap)

    # FULL: windowed leaf, whole-subtree coverage, no sidecar (D14).
    full_cov = {
        "spec": "morton-moc/1",
        "box": [FULL, None, None, None],
        "cell_order": 8,
        "source": "worker",
        "encoding": "full",
    }
    full_leaf = root / convention.leaf_path(FULL, window="2019")
    _write_json(
        full_leaf / "zarr.json",
        _zarr_group({convention.COMMIT_ATTR: _stamp(full_cov, window="2019")}),
    )

    # DEBRIS: a leaf prefix whose root metadata lacks the stamp.
    _write_json(root / convention.leaf_path(DEBRIS) / "zarr.json", _zarr_group({}))

    # A foreign object at the root the walk must ignore (root-only exception
    # names are objects, not digit children).
    _write_json(root / "notes.json", {"scratch": True})

    return str(root)

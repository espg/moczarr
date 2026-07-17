"""Shared fixture: a hand-built local hive store matching the mortie#62 layout.

Built object-by-object (no zagg dependency) so the tests document the wire
format explicitly; the byte-level conventions are pinned by the golden
vectors in ``test_coverage.py``.
"""

import json

import pytest
from numcodecs import Zstd

from moczarr import convention

#: The three shards in the fixture store (order 6, southern base -5).
STAMPED = "-5112333"  # bitmap-encoding leaf, occupied cell = shard + "11"
FULL = "-5112331"  # windowed leaf (2019), encoding "full"
DEBRIS = "-5112334"  # leaf prefix without a commit stamp


def _manifest():
    return {
        "spec": convention.HIVE_SPEC,
        "dataset": {"short_name": "ATL06", "version": "007"},
        "cell_order": 8,
        "shard_order": 6,
        "split_schedule": [1] * 6,
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

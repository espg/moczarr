"""Concurrent metadata reads (issue #5, phase 2) over a synthetic many-leaf store.

The concurrency contract: identical RESULTS to the serial path — same
stamps, same leaf set, same final dataset — with the request batching an
implementation detail. Local-fs timing shows none of the S3 latency win, so
the wall-clock check here is only a coarse "not pathologically slower"
bound; the measured win lives in ``tools/bench_open.py``.
"""

import asyncio
import itertools
import time

import numpy as np
import pytest
from conftest import build_many_leaf_store

from moczarr import convention, open_hive, store

#: 220 order-6 shards under one order-2 subtree (4^4 = 256 available).
SHARDS = ["-511" + "".join(p) for p in itertools.product("1234", repeat=4)][:220]


@pytest.fixture(scope="module")
def many_leaf_store(tmp_path_factory):
    return build_many_leaf_store(tmp_path_factory.mktemp("many") / "store", SHARDS)


class TestReadCommits:
    def test_concurrent_matches_serial(self, many_leaf_store):
        leaves = [convention.leaf_path(s) for s in SHARDS]
        serial = store.read_commits(many_leaf_store, leaves, concurrency=1)
        batched = store.read_commits(many_leaf_store, leaves, concurrency=32)
        assert batched == serial
        assert all(stamp["complete"] is True for stamp in batched)

    def test_absent_and_debris_read_none(self, many_leaf_store):
        # Position-aligned None for a never-written shard, same as read_commit.
        leaves = [convention.leaf_path(SHARDS[0]), convention.leaf_path("-544444")]
        stamps = store.read_commits(many_leaf_store, leaves, concurrency=8)
        assert stamps[0] is not None and stamps[1] is None


class TestWalkConcurrency:
    def test_same_set_as_serial(self, many_leaf_store):
        serial = sorted(store.walk_leaves(many_leaf_store))
        breadth = sorted(store.walk_leaves(many_leaf_store, concurrency=16))
        assert breadth == serial
        assert len(serial) == len(SHARDS)


class TestOpenHiveConcurrency:
    def test_concurrent_equals_serial(self, many_leaf_store):
        ds_serial = open_hive(many_leaf_store, concurrency=1)
        ds_batched = open_hive(many_leaf_store, concurrency=32)
        assert ds_batched.sizes["cells"] == len(SHARDS) * 16
        np.testing.assert_array_equal(ds_batched["morton"].values, ds_serial["morton"].values)
        np.testing.assert_array_equal(ds_batched["count"].values, ds_serial["count"].values)

    def test_walk_fallback_concurrent_equals_serial(self, tmp_path):
        # No root MOC -> the discovery walk feeds the same result either way.
        root = build_many_leaf_store(tmp_path / "store", SHARDS[:64])
        (tmp_path / "store" / convention.ROOT_COVERAGE_NAME).unlink()
        ds_serial = open_hive(root, concurrency=1)
        ds_batched = open_hive(root, concurrency=32)
        np.testing.assert_array_equal(ds_batched["morton"].values, ds_serial["morton"].values)

    def test_wall_clock_sanity(self, many_leaf_store):
        # Local fs shows none of the S3 latency win; this only guards against
        # the batched path being pathologically slower. Do NOT tighten — a
        # sharper timing assertion would be flaky.
        t0 = time.perf_counter()
        open_hive(many_leaf_store, concurrency=1)
        serial = time.perf_counter() - t0
        t0 = time.perf_counter()
        open_hive(many_leaf_store, concurrency=32)
        batched = time.perf_counter() - t0
        assert batched <= serial * 1.5

    def test_constructions_still_o1(self, many_leaf_store, monkeypatch):
        # The phase-1 invariant survives batching: one obstore store, total.
        import moczarr.open as open_module
        from moczarr import store as store_module

        counts = {"obstore": 0}
        real_open = store_module.open_object_store

        def counting_open(path, **kwargs):
            counts["obstore"] += 1
            return real_open(path, **kwargs)

        monkeypatch.setattr(store_module, "open_object_store", counting_open)
        monkeypatch.setattr(open_module, "open_object_store", counting_open)
        open_hive(many_leaf_store, concurrency=32)
        assert counts["obstore"] == 1


class TestJupyterSafety:
    def test_open_hive_inside_running_loop(self, many_leaf_store):
        # The notebook case: open_hive called while an event loop is running
        # (asyncio.run would raise); the runner hops to a worker thread.
        async def notebook_cell():
            return open_hive(many_leaf_store, aoi=[SHARDS[0]], concurrency=8)

        ds = asyncio.run(notebook_cell())
        assert ds.sizes["cells"] == 16

    def test_run_coroutine_both_contexts(self):
        async def value():
            return 42

        assert store._run_coroutine(value()) == 42  # no loop running

        async def nested():
            return store._run_coroutine(value())  # loop running -> thread hop

        assert asyncio.run(nested()) == 42

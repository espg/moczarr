"""Measure the issue #5 shared-handle + concurrency win on a synthetic store.

Builds a many-leaf local hive store (the ``tests/conftest.py`` builder,
imported by path — moczarr-only, no zagg) and reports, per ``open_hive``
posture:

- obstore store constructions (the phase-1 O(1) acceptance),
- object-store request counts (GETs and delimiter-LISTs),
- wall clock, serial (``concurrency=1``) vs batched (``concurrency=32``),

plus the phase-3 MEASUREMENT arm espg ratified on the issue: thread-pooled
per-leaf DATA opens (``xr.open_zarr`` over the shared wrapper in a
``ThreadPoolExecutor``) vs the serial loop — numbers only, no library
change; the decision waits on the phase-5 lazy-index work.

Local-fs latency is microseconds, so the concurrency numbers here bound the
*overhead* of batching, not the S3 win (that is ~RTT-bound: N serial GETs
vs ceil(N/32) batched rounds — see the issue #5 trade-space table)::

    uv run python tools/bench_open.py [--leaves 220] [--concurrency 32]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from conftest import build_many_leaf_store  # noqa: E402

import moczarr.open as open_module  # noqa: E402
from moczarr import convention, open_hive, store  # noqa: E402


class Counters:
    """Monkeypatch-style wrappers counting constructions and requests."""

    def __init__(self):
        self.constructions = 0
        self.gets = 0
        self.lists = 0

    def install(self):
        import obstore

        real_open = store.open_object_store
        real_get, real_get_async = obstore.get, obstore.get_async
        real_list, real_list_async = (
            obstore.list_with_delimiter,
            obstore.list_with_delimiter_async,
        )

        def counting_open(path, **kwargs):
            self.constructions += 1
            return real_open(path, **kwargs)

        def counting_get(*args, **kwargs):
            self.gets += 1
            return real_get(*args, **kwargs)

        def counting_get_async(*args, **kwargs):
            self.gets += 1
            return real_get_async(*args, **kwargs)

        def counting_list(*args, **kwargs):
            self.lists += 1
            return real_list(*args, **kwargs)

        def counting_list_async(*args, **kwargs):
            self.lists += 1
            return real_list_async(*args, **kwargs)

        store.open_object_store = counting_open
        open_module.open_object_store = counting_open
        obstore.get, obstore.get_async = counting_get, counting_get_async
        obstore.list_with_delimiter = counting_list
        obstore.list_with_delimiter_async = counting_list_async
        # zarr's ObjectStore adapter resolves obstore functions off the
        # module at call time, so leaf DATA reads issued by zarr ARE counted
        # (measured: 220 leaves -> 6 GETs/leaf + manifest + MOC, and one
        # member-discovery LIST per leaf group), alongside the metadata
        # tiers (manifest, MOC, stamps, walk).

    def reset(self):
        self.constructions = self.gets = self.lists = 0


def bench_open(root: str, counters: Counters, concurrency: int | None) -> dict:
    counters.reset()
    t0 = time.perf_counter()
    ds = open_hive(root, concurrency=concurrency)
    elapsed = time.perf_counter() - t0
    return {
        "cells": ds.sizes["cells"],
        "constructions": counters.constructions,
        "gets": counters.gets,
        "lists": counters.lists,
        "wall_s": elapsed,
    }


def bench_threaded_leaf_opens(root: str, shards: list[str], workers: int) -> dict:
    """The ratified phase-3 measurement: thread-pooled leaf DATA opens.

    Serial loop vs ThreadPoolExecutor over per-leaf ``xr.open_zarr`` on ONE
    shared wrapper — the experiment lives here only; ``open_hive`` keeps its
    serial loop until the numbers (and the phase-5 lazy-index work) justify
    the xarray thread-safety surface.
    """
    from concurrent.futures import ThreadPoolExecutor

    import xarray as xr
    from zarr.storage import ObjectStore

    wrapper = ObjectStore(store.open_object_store(root), read_only=True)
    manifest = store.read_manifest(root)
    groups = [f"{convention.leaf_path(s)}/{manifest['cell_order']}" for s in shards]

    def open_one(group):
        return xr.open_zarr(wrapper, group=group, consolidated=False, zarr_format=3)

    t0 = time.perf_counter()
    serial = [open_one(g) for g in groups]
    serial_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        threaded = list(pool.map(open_one, groups))
    threaded_s = time.perf_counter() - t0
    assert len(serial) == len(threaded) == len(groups)
    return {"serial_s": serial_s, "threaded_s": threaded_s, "workers": workers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--leaves", type=int, default=220)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8, help="leaf-open thread pool size")
    args = parser.parse_args()

    shards = ["".join(("-511", *p)) for p in itertools.product("1234", repeat=4)]
    shards = (shards * (args.leaves // len(shards) + 1))[: args.leaves]
    shards = list(dict.fromkeys(shards))  # unique, order kept (caps at 256)

    counters = Counters()
    counters.install()
    with tempfile.TemporaryDirectory() as tmp:
        root = build_many_leaf_store(Path(tmp) / "store", shards)
        rows = [
            ("serial (concurrency=1)", bench_open(root, counters, 1)),
            (
                f"batched (concurrency={args.concurrency})",
                bench_open(root, counters, args.concurrency),
            ),
        ]
        print(f"\nopen_hive over {len(shards)} leaves (local fs):\n")
        print("| posture | store constructions | GETs | LISTs | wall clock |")
        print("|---|---|---|---|---|")
        for name, r in rows:
            print(
                f"| {name} | {r['constructions']} | {r['gets']} | {r['lists']} "
                f"| {r['wall_s']:.2f} s |"
            )
        leaf = bench_threaded_leaf_opens(root, shards, args.workers)
        print(f"\nleaf DATA opens, measurement only ({len(shards)} leaves):\n")
        print("| leaf-open strategy | wall clock |")
        print("|---|---|")
        print(f"| serial loop | {leaf['serial_s']:.2f} s |")
        print(f"| ThreadPoolExecutor({leaf['workers']}) | {leaf['threaded_s']:.2f} s |")
        speedup = leaf["serial_s"] / leaf["threaded_s"] if leaf["threaded_s"] else float("inf")
        print(f"\nthreaded leaf-open speedup (local fs): {speedup:.2f}x")


if __name__ == "__main__":
    main()

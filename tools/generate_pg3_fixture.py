"""Generate the committed ``path_grouping: 3`` golden fixture (moczarr-built).

Run from the moczarr environment::

    uv run python tools/generate_pg3_fixture.py --out tests/data/serc_hive_pg3

zagg's writer declares ``path_grouping`` in the manifest (D21) but still
delegates path construction to mortie's one-digit ``hive_path``, so it
cannot yet WRITE a grouped store — this conftest-style raw-object builder
substitutes until it can (the zagg-written replacement is tracked on
espg/moczarr#11). Every byte is deterministic (fixed shards, fixed values,
fixed timestamps), so regeneration is reproducible.

Layout (spec §6.1, espg/mortie#62): three order-8 shards — two northern
(SERC area, one contiguous root-MOC range) and one southern — with
``cell_order`` 10 and ``path_grouping: 3``, so the digit tail chunks
``3+3+2``: the committed tree exercises BOTH full-width components and the
short remainder component the spec's chunking rule produces when the shard
order does not divide by the grouping. Leaves are fully occupied
(``encoding: "full"``, no sidecar): ``morton`` holds all 16 subtree cells in
ascending packed order; ``count`` is ``1..16``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from moczarr import convention

#: Deterministic order-8 shards (geo2mort of fixed SERC-area/antipode points).
SHARDS = ["433142241", "433142242", "-433412214"]
PATH_GROUPING = 3
SHARD_ORDER = 8
CELL_ORDER = 10
GENERATED_AT = "2026-07-20T00:00:00+00:00"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def _array_meta(data_type: str, length: int) -> dict:
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


def build(out: Path) -> None:
    import itertools

    if out.exists():
        shutil.rmtree(out)
    depth = CELL_ORDER - SHARD_ORDER
    suffixes = ["".join(p) for p in itertools.product("1234", repeat=depth)]
    _write_json(
        out / convention.MANIFEST_NAME,
        {
            "spec": convention.HIVE_SPEC,
            "dataset": {"short_name": "ATL06", "version": "007"},
            "cell_order": CELL_ORDER,
            "shard_order": SHARD_ORDER,
            "split_schedule": [1] * SHARD_ORDER,
            "path_grouping": PATH_GROUPING,
            "pyramid": {"orders": [], "aggregation": {}},
            "generated_at": GENERATED_AT,
        },
    )
    _write_json(
        out / convention.ROOT_COVERAGE_NAME,
        {
            "spec": "morton-moc/1",
            "encoding": "ranges",
            "order": SHARD_ORDER,
            "source": "builder",
            "generated_at": GENERATED_AT,
            # The two northern shards are rank-consecutive: one range.
            "ranges": [[SHARDS[0], SHARDS[1]], [SHARDS[2], SHARDS[2]]],
        },
    )
    group_meta = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    n = len(suffixes)
    count_bytes = np.arange(1, n + 1, dtype="<i8").tobytes()
    for shard in SHARDS:
        stamp = {
            "spec": convention.HIVE_SPEC,
            "complete": True,
            "cells_with_data": n,
            "granule_count": 1,
            "written_at": GENERATED_AT,
            "coverage": {
                "spec": "morton-moc/1",
                "box": [shard, None, None, None],
                "cell_order": CELL_ORDER,
                "source": "builder",
                "encoding": "full",
            },
        }
        leaf = out / convention.leaf_path(shard, path_grouping=PATH_GROUPING)
        _write_json(
            leaf / "zarr.json",
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {convention.COMMIT_ATTR: stamp},
            },
        )
        words = np.sort(
            np.asarray([convention.morton_word(shard + s) for s in suffixes], dtype=np.uint64)
        )
        group_dir = leaf / str(CELL_ORDER)
        for name, meta, chunk in (
            ("morton", json.dumps(_array_meta("uint64", n)), words.astype("<u8").tobytes()),
            ("count", json.dumps(_array_meta("int64", n)), count_bytes),
        ):
            (group_dir / name / "c").mkdir(parents=True)
            (group_dir / name / "zarr.json").write_text(meta)
            (group_dir / name / "c" / "0").write_bytes(chunk)
        (group_dir / "zarr.json").write_text(group_meta)
    n_files = sum(1 for f in out.rglob("*") if f.is_file())
    size_kb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024
    print(f"wrote {out}: {n_files} objects, {size_kb:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)

"""Generate the committed multi-product golden fixture (moczarr-built).

Run from the moczarr environment::

    uv run python tools/generate_multiproduct_fixture.py --out tests/data/multiproduct_hive

zagg's writer targets one product tree per run and has no multi-product
STORE-root builder to borrow (a D19 store root is just a directory of
independently written products), so this conftest-style raw-object builder
composes one deterministically — every byte fixed (fixed shards, values,
timestamps), so regeneration is reproducible. The per-product bytes follow
the same wire format the zagg-written SERC fixture pins.

Layout (zagg D19 / mortie spec §6.5): a store root carrying ONLY name-shaped
children — no root manifest (the §6.5 content discrimination) — with:

- ``atl06/`` — a ``morton-hive/1`` product: two order-3 shards
  (``morton`` + ``count``), ``aggregation.yaml`` present and the manifest's
  ``semantic_hash`` = sha256 of its bytes (the D19 self-consistency), D20
  ``stats.json`` sidecars carrying O11 ``content_hashes`` computed from the
  written array bytes, and D22 ``stats.rollup.json`` objects at the shard
  nodes and every ancestor (hand-folded with the D20 merge dispositions).
- ``atl06_windows/`` — a ``morton-hive/2`` windowed product (``morton`` +
  ``height``), no ``aggregation.yaml`` and no ``semantic_hash`` (the pre-D19
  surface), ``stats_{window}.json`` sidecars: one WITH content hashes, one
  without (``match: None``), one leaf with no sidecar at all.
- ``scratch/`` — a name-shaped child WITHOUT a manifest (not a product;
  ``list_products`` must skip it).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from moczarr import convention, stats

SHARD_ORDER = 3
CELL_ORDER = 5
GENERATED_AT = "2026-07-27T00:00:00+00:00"
RUN_ID = "fixture-run"

AGGREGATION_YAML = """\
# canonical semantic core (D19) — deterministic fixture bytes
dataset: {short_name: ATL06, version: '007'}
aggregation: {count: {func: count}}
"""


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


def _record(shard: str, *, window, n_obs: int, timestamp: str, semantic_hash, content_hashes):
    """One D20 stats record (schema_version 1), deterministic fields only."""
    record = {
        "schema_version": 1,
        "shard_key": convention.morton_word(shard),
        "window": window,
        "run_id": RUN_ID,
        "semantic_hash": semantic_hash,
        "zagg_version": "0.36.0",
        "n_shards": 1,
        "n_granules": 1,
        "granules_sha256": hashlib.sha256(f"granule-{shard}-{window}".encode()).hexdigest(),
        "n_obs": n_obs,
        "cells_with_data": 16,
        "phase_timings": {"read": 1.0, "aggregate": 0.5, "write": 0.25},
        "duration_s": 2.0,
        "spill_bytes": None,
        "gb_seconds": None,
        "est_cost_usd": None,
        "max_memory_mb": 512.0,
        "container_hwm_mb": None,
        "lambda": None,
        "timestamp": timestamp,
        "success": True,
        "error": None,
        "invoked_by": None,
    }
    if content_hashes is not None:
        record["content_hashes"] = content_hashes
    return record


def _merge(records: list[dict]) -> dict:
    """The D20 fold (mirrors ``zagg.telemetry.merge`` for the keys present)."""
    eq_keys = (
        "window",
        "shard_key",
        "run_id",
        "semantic_hash",
        "granules_sha256",
        "zagg_version",
        "lambda",
        "invoked_by",
        "error",
    )
    out: dict = {"schema_version": 1}
    for key in eq_keys:
        first = records[0].get(key)
        out[key] = first if all(r.get(key) == first for r in records) else None
    for key in ("n_shards", "n_granules", "n_obs", "cells_with_data", "duration_s"):
        out[key] = sum(r.get(key) or 0 for r in records)
    timings: dict[str, float] = {}
    for r in records:
        for name, secs in (r.get("phase_timings") or {}).items():
            timings[name] = timings.get(name, 0.0) + secs
    out["phase_timings"] = timings
    for key in ("gb_seconds", "est_cost_usd", "spill_bytes"):
        vals = [r.get(key) for r in records if r.get(key) is not None]
        out[key] = sum(vals) if vals else None
    for key in ("max_memory_mb", "container_hwm_mb"):
        vals = [r.get(key) for r in records if r.get(key) is not None]
        out[key] = max(vals) if vals else None
    out["timestamp"] = max(r["timestamp"] for r in records)
    out["success"] = all(r["success"] for r in records)
    return out


def _write_leaf(root: Path, shard: str, arrays: dict[str, np.ndarray], *, window, spec):
    """One raw-object leaf zarr (stamped, ``encoding: "full"``); returns hashes."""
    stamp = {
        "spec": spec,
        "complete": True,
        "cells_with_data": 16,
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
    if window is not None:
        stamp["window"] = window
    leaf = root / convention.leaf_path(shard, window=window)
    _write_json(
        leaf / "zarr.json",
        {"zarr_format": 3, "node_type": "group", "attributes": {convention.COMMIT_ATTR: stamp}},
    )
    group_dir = leaf / str(CELL_ORDER)
    dtypes = {"uint64": "<u8", "int64": "<i8", "float64": "<f8"}
    hashes = {}
    for name, values in arrays.items():
        data_type = str(values.dtype)
        chunk = values.astype(dtypes[data_type]).tobytes()
        (group_dir / name / "c").mkdir(parents=True)
        (group_dir / name / "zarr.json").write_text(json.dumps(_array_meta(data_type, len(values))))
        (group_dir / name / "c" / "0").write_bytes(chunk)
        hashes[f"{CELL_ORDER}/{name}"] = hashlib.sha256(chunk).hexdigest()
    (group_dir / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    return hashes


def _cells(shard: str) -> np.ndarray:
    suffixes = itertools.product("1234", repeat=CELL_ORDER - SHARD_ORDER)
    words = [convention.morton_word(shard + "".join(s)) for s in suffixes]
    return np.sort(np.asarray(words, dtype=np.uint64))


def _manifest(spec: str, **extra) -> dict:
    return {
        "spec": spec,
        "dataset": {"short_name": "ATL06", "version": "007"},
        "cell_order": CELL_ORDER,
        "shard_order": SHARD_ORDER,
        "split_schedule": [1] * SHARD_ORDER,
        "path_grouping": 1,
        "pyramid": {"orders": [], "aggregation": {}},
        "generated_at": GENERATED_AT,
        **extra,
    }


def _root_coverage(shards: list[str]) -> dict:
    return {
        "spec": "morton-moc/1",
        "encoding": "ranges",
        "order": SHARD_ORDER,
        "source": "builder",
        "generated_at": GENERATED_AT,
        "ranges": [[s, s] for s in shards],
    }


def _rollup(node: str, payload: dict, generation: dict, *, windows=None) -> dict:
    envelope = {
        "spec": stats.SWEEP_SPEC,
        "family": "stats",
        "node": node,
        "order": convention.decimal_order(node),
        "generation": generation,
    }
    if windows is not None:
        envelope["windows"] = windows
    envelope["payload"] = payload
    return envelope


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)

    # -- atl06: /1, aggregation.yaml + semantic_hash, sidecars w/ O11, rollups
    product = out / "atl06"
    semantic_hash = hashlib.sha256(AGGREGATION_YAML.encode()).hexdigest()
    shards = ["4111", "4112"]
    _write_json(
        product / convention.MANIFEST_NAME,
        _manifest(convention.HIVE_SPEC, semantic_hash=semantic_hash),
    )
    _write_json(product / convention.ROOT_COVERAGE_NAME, _root_coverage(shards))
    (product / "aggregation.yaml").write_text(AGGREGATION_YAML)
    records = {}
    for i, shard in enumerate(shards):
        arrays = {"morton": _cells(shard), "count": np.arange(1, 17, dtype=np.int64) * (i + 1)}
        hashes = _write_leaf(product, shard, arrays, window=None, spec=convention.HIVE_SPEC)
        record = _record(
            shard,
            window=None,
            n_obs=100 * (i + 1),
            timestamp=f"2026-07-27T00:0{i}:00+00:00",
            semantic_hash=semantic_hash,
            content_hashes={"arrays": hashes, "combined": stats.combined_hash(hashes)},
        )
        records[shard] = record
        leaf = convention.leaf_path(shard)
        _write_json(product / stats.stats_sidecar_path(leaf, convention.HIVE_SPEC), record)
    # D22 rollups: shard nodes (windows key) and every ancestor up to the base.
    for shard in shards:
        _write_json(
            product / stats._node_rel(shard, 1) / stats.STATS_ROLLUP_NAME,
            _rollup(
                shard,
                _merge([records[shard]]),
                {"n_leaves": 1, "max_leaf_timestamp": records[shard]["timestamp"]},
                windows=[None],
            ),
        )
    merged = _merge(list(records.values()))
    generation = {
        "n_leaves": 2,
        "max_leaf_timestamp": max(r["timestamp"] for r in records.values()),
    }
    for node in ("411", "41", "4"):
        _write_json(
            product / stats._node_rel(node, 1) / stats.STATS_ROLLUP_NAME,
            _rollup(node, merged, generation),
        )

    # -- atl06_windows: /2, no aggregation.yaml / semantic_hash, mixed sidecars
    product = out / "atl06_windows"
    shards = ["-5111", "-5112"]
    _write_json(
        product / convention.MANIFEST_NAME,
        _manifest(convention.HIVE_SPEC_V2, temporal={"schedule": "yearly"}),
    )
    _write_json(product / convention.ROOT_COVERAGE_NAME, _root_coverage(shards))
    for shard, window, n_obs, content in (
        ("-5111", "2019", 10, True),  # sidecar WITH content hashes
        ("-5111", "2020", 20, False),  # sidecar WITHOUT content hashes
        ("-5112", "2019", 30, None),  # no sidecar at all
    ):
        seed = float(n_obs)
        arrays = {
            "morton": _cells(shard),
            "height": np.linspace(seed, seed + 1.5, 16, dtype=np.float64),
        }
        hashes = _write_leaf(product, shard, arrays, window=window, spec=convention.HIVE_SPEC_V2)
        if content is None:
            continue
        record = _record(
            shard,
            window=window,
            n_obs=n_obs,
            timestamp="2026-07-27T00:03:00+00:00",
            semantic_hash=None,
            content_hashes=(
                {"arrays": hashes, "combined": stats.combined_hash(hashes)} if content else None
            ),
        )
        leaf = convention.leaf_path(shard, window=window)
        _write_json(product / stats.stats_sidecar_path(leaf, convention.HIVE_SPEC_V2), record)

    # -- scratch: name-shaped child with no manifest (not a product).
    _write_json(out / "scratch" / "notes.json", {"scratch": True})

    n_files = sum(1 for f in out.rglob("*") if f.is_file())
    size_kb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024
    print(f"wrote {out}: {n_files} objects, {size_kb:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)

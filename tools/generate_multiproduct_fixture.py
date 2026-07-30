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
  written array bytes (independently of ``stats.hash_arrays``), and D22
  ``stats.rollup.json`` objects at the shard nodes and every ancestor
  (hand-folded with the D20 merge dispositions). The COMBINED digest here
  does come from ``stats.combined_hash``, so it is self-consistent by
  construction — the serialization itself is pinned by the
  ``FROZEN_COMBINED_4111`` literal in ``tests/test_stats.py``, which is what
  a change of joiner/keying has to get past.
- ``atl06_windows/`` — a ``morton-hive/2`` windowed product (``morton`` +
  ``height``), no ``aggregation.yaml`` and no ``semantic_hash`` (the pre-D19
  surface), ``stats_{window}.json`` sidecars: one WITH content hashes, one
  without (``match: None``), one leaf with no sidecar at all.
- ``atl06_ragged/`` — a ``morton-hive/1`` product carrying the RAGGED (vlen)
  O11 surface: a ``variable_length_bytes`` digest payload array plus its
  ``{field}_locations`` uint64-bytes sibling (zagg issues #209/#87), so the
  length-prefixed vlen hash recipe (``stats.hash_arrays``) is pinned by a
  golden rather than untested. The vlen chunks are written raw here — the
  zarr ``vlen-bytes`` wire form is ``uint32_le`` element count then
  ``uint32_le`` length + bytes per element — with NO zstd (zagg's chain adds
  it; O11 hashes decoded values, so the digest is identical either way and
  the fixture stays byte-reproducible).
- ``atl06_pg3/`` — a ``morton-hive/1`` product at ``path_grouping: 3`` (D21):
  ONE order-5 shard, so the digit tail chunks 3+2 (a full-width component and
  the short remainder), with a sidecar and rollups at the component-boundary
  nodes — the grouped sidecar/rollup key arithmetic exercised, not only
  theoretically supported.
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
from typing import NamedTuple

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


class Vlen(NamedTuple):
    """A ragged (vlen-bytes) array: one raw little-endian payload per cell."""

    payloads: list[bytes]
    element_dtype: str
    locations: str | None = None


def _vlen_array_meta(length: int, *, element_dtype: str, locations: str | None) -> dict:
    """Uncompressed ``variable_length_bytes`` metadata (vlen-bytes codec only).

    Carries zagg's ``ragged`` attrs block (``grids.base.RAGGED_ELEMENT_ATTR``)
    so the fixture matches the on-disk form; O11 hashes VALUES, so the attrs
    are not part of the digest.
    """
    meta = _array_meta("uint64", length)
    meta["data_type"] = "variable_length_bytes"
    meta["fill_value"] = ""
    meta["codecs"] = [{"name": "vlen-bytes", "configuration": {}}]
    ragged = {"spec": "zagg-ragged/1", "element": {"dtype": element_dtype, "shape": [-1]}}
    if locations is not None:
        ragged["locations"] = locations
    meta["attributes"] = {"ragged": ragged}
    return meta


def _vlen_chunk(payloads: list[bytes]) -> bytes:
    """The zarr ``vlen-bytes`` chunk wire form (count, then len+bytes each)."""
    body = b"".join(len(p).to_bytes(4, "little") + p for p in payloads)
    return len(payloads).to_bytes(4, "little") + body


def _vlen_hash(payloads: list[bytes]) -> str:
    """The O11 vlen recipe, hand-rolled: sha256 of uint64_le(len)||payload, in order.

    Deliberately independent of ``stats.hash_arrays`` (which reads the same
    bytes back through zarr) so the fixture golden-tests the recipe.
    """
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _record(
    shard: str,
    *,
    window,
    n_obs: int,
    timestamp: str,
    semantic_hash,
    content_hashes,
    cells_with_data: int = 16,
):
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
        "cells_with_data": cells_with_data,
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


def _write_leaf(
    root: Path,
    shard: str,
    arrays: dict,
    *,
    window,
    spec,
    cell_order: int = CELL_ORDER,
    path_grouping: int = 1,
):
    """One raw-object leaf zarr (stamped, ``encoding: "full"``); returns hashes.

    An ``arrays`` value that is a :class:`Vlen` (rather than an ndarray) is
    written as a ``variable_length_bytes`` vlen array — one payload per cell.
    """
    stamp = {
        "spec": spec,
        "complete": True,
        "cells_with_data": len(next(iter(arrays.values()))),
        "granule_count": 1,
        "written_at": GENERATED_AT,
        "coverage": {
            "spec": "morton-moc/1",
            "box": [shard, None, None, None],
            "cell_order": cell_order,
            "source": "builder",
            "encoding": "full",
        },
    }
    if window is not None:
        stamp["window"] = window
    leaf = root / convention.leaf_path(shard, window=window, path_grouping=path_grouping)
    _write_json(
        leaf / "zarr.json",
        {"zarr_format": 3, "node_type": "group", "attributes": {convention.COMMIT_ATTR: stamp}},
    )
    group_dir = leaf / str(cell_order)
    dtypes = {"uint64": "<u8", "int64": "<i8", "float64": "<f8"}
    hashes = {}
    for name, values in arrays.items():
        (group_dir / name / "c").mkdir(parents=True)
        if isinstance(values, Vlen):  # vlen (ragged) array: one payload per cell
            (group_dir / name / "zarr.json").write_text(
                json.dumps(
                    _vlen_array_meta(
                        len(values.payloads),
                        element_dtype=values.element_dtype,
                        locations=values.locations,
                    )
                )
            )
            (group_dir / name / "c" / "0").write_bytes(_vlen_chunk(values.payloads))
            hashes[f"{cell_order}/{name}"] = _vlen_hash(values.payloads)
            continue
        data_type = str(values.dtype)
        chunk = values.astype(dtypes[data_type]).tobytes()
        (group_dir / name / "zarr.json").write_text(json.dumps(_array_meta(data_type, len(values))))
        (group_dir / name / "c" / "0").write_bytes(chunk)
        hashes[f"{cell_order}/{name}"] = hashlib.sha256(chunk).hexdigest()
    (group_dir / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    return hashes


def _cells(shard: str, *, cell_order: int = CELL_ORDER) -> np.ndarray:
    suffixes = itertools.product("1234", repeat=cell_order - convention.decimal_order(shard))
    words = [convention.morton_word(shard + "".join(s)) for s in suffixes]
    return np.sort(np.asarray(words, dtype=np.uint64))


def _manifest(
    spec: str,
    *,
    cell_order: int = CELL_ORDER,
    shard_order: int = SHARD_ORDER,
    path_grouping: int = 1,
    **extra,
) -> dict:
    return {
        "spec": spec,
        "dataset": {"short_name": "ATL06", "version": "007"},
        "cell_order": cell_order,
        "shard_order": shard_order,
        "split_schedule": [1] * shard_order,
        "path_grouping": path_grouping,
        "pyramid": {"orders": [], "aggregation": {}},
        "generated_at": GENERATED_AT,
        **extra,
    }


def _root_coverage(shards: list[str], *, shard_order: int = SHARD_ORDER) -> dict:
    return {
        "spec": "morton-moc/1",
        "encoding": "ranges",
        "order": shard_order,
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

    # -- atl06_ragged: /1 with the vlen (ragged) O11 surface, one shard.
    product = out / "atl06_ragged"
    shard = "4111"
    _write_json(product / convention.MANIFEST_NAME, _manifest(convention.HIVE_SPEC))
    _write_json(product / convention.ROOT_COVERAGE_NAME, _root_coverage([shard]))
    # Cell i carries i float32 quantiles and i uint64 location words — cell 0
    # is EMPTY, so the zero-length prefix case is pinned too.
    digest_payloads = [np.arange(i, dtype="<f4").tobytes() for i in range(16)]
    location_payloads = [np.arange(i, dtype="<u8").tobytes() for i in range(16)]
    arrays = {
        "morton": _cells(shard),
        "h_li_tdigest": Vlen(digest_payloads, "float32", "h_li_tdigest_locations"),
        "h_li_tdigest_locations": Vlen(location_payloads, "uint64"),
    }
    hashes = _write_leaf(product, shard, arrays, window=None, spec=convention.HIVE_SPEC)
    record = _record(
        shard,
        window=None,
        n_obs=120,
        timestamp="2026-07-27T00:04:00+00:00",
        semantic_hash=None,
        content_hashes={"arrays": hashes, "combined": stats.combined_hash(hashes)},
    )
    _write_json(
        product / stats.stats_sidecar_path(convention.leaf_path(shard), convention.HIVE_SPEC),
        record,
    )

    # -- atl06_pg3: /1 at path_grouping 3 (D21), so the sidecar + rollup key
    # arithmetic is exercised on a GROUPED store, not only theoretically
    # supported. Order-5 shard, so the digit tail chunks 3+2 — both a
    # full-width component and the short remainder.
    product = out / "atl06_pg3"
    shard = "411121"
    cell_order, shard_order, grouping = 6, 5, 3
    _write_json(
        product / convention.MANIFEST_NAME,
        _manifest(
            convention.HIVE_SPEC,
            cell_order=cell_order,
            shard_order=shard_order,
            path_grouping=grouping,
        ),
    )
    _write_json(
        product / convention.ROOT_COVERAGE_NAME,
        _root_coverage([shard], shard_order=shard_order),
    )
    arrays = {
        "morton": _cells(shard, cell_order=cell_order),
        "count": np.arange(1, 5, dtype=np.int64),
    }
    hashes = _write_leaf(
        product,
        shard,
        arrays,
        window=None,
        spec=convention.HIVE_SPEC,
        cell_order=cell_order,
        path_grouping=grouping,
    )
    record = _record(
        shard,
        window=None,
        n_obs=40,
        timestamp="2026-07-27T00:05:00+00:00",
        semantic_hash=None,
        content_hashes={"arrays": hashes, "combined": stats.combined_hash(hashes)},
        cells_with_data=4,
    )
    leaf = convention.leaf_path(shard, path_grouping=grouping)
    _write_json(product / stats.stats_sidecar_path(leaf, convention.HIVE_SPEC), record)
    # Rollups at the COMPONENT-BOUNDARY nodes only — the grouped tree's real
    # directory levels (``411121`` -> 4/111/21, ``4111`` -> 4/111, ``4`` -> 4).
    generation = {"n_leaves": 1, "max_leaf_timestamp": record["timestamp"]}
    for node, windows in ((shard, [None]), ("4111", None), ("4", None)):
        _write_json(
            product / stats._node_rel(node, grouping) / stats.STATS_ROLLUP_NAME,
            _rollup(node, _merge([record]), generation, windows=windows),
        )

    # -- scratch: name-shaped child with no manifest (not a product).
    _write_json(out / "scratch" / "notes.json", {"scratch": True})

    n_files = sum(1 for f in out.rglob("*") if f.is_file())
    size_kb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024
    print(f"wrote {out}: {n_files} objects, {size_kb:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)

"""Generate the committed overview-pyramid fixture with zagg's REAL writer.

Run from a zagg checkout's environment (zagg is deliberately not a moczarr
dependency)::

    cd ../zagg && uv run python ../moczarr/tools/generate_overview_fixture.py \
        --out ../moczarr/tests/data/overview_hive

Every byte comes from zagg's own write path — ``build_manifest`` /
``ensure_manifest`` (the template-time ``pyramid`` declaration,
``zagg.sweep_overview.build_pyramid_block``), ``process_and_write_hive`` per
source leaf, ``write_root_coverage``, and then the D22 second-pass overview
sweep (``zagg.sweep_overview.sweep_overviews``) — the ratified zagg-real-writer
posture (``generate_serc_fixture.py``, espg/moczarr#1), so writer<->reader
drift fails moczarr's suite on whichever side moved. The exact zagg revision
is recorded in the golden sidecar (``overview_hive.golden.json``).

Layout (a D19 multi-product store root; SERC-area shards, deliberately tiny —
parent order 6 / child order 8, 16 cells per leaf):

- ``atl06/`` — ``morton-hive/1``, the default atl06 aggregation (no ragged
  fields), ``output.pyramid: {orders: [4, 2]}`` → overview cell orders 6 and
  4 under the constant-depth rule (spec §4.4: ``k_cell = c - (s - k)``). The
  D24 field classes split naturally: ``count``/``h_min``/``h_max`` are exact
  (sum/min/min laws) and roll up; ``h_mean``/``h_sigma``/``h_variance``/
  ``h_q25``/``h_q50``/``h_q75`` are ``none``-class and exist ONLY at native
  resolution (option A, the ruled default — the recorded absence lives in the
  manifest's ``pyramid.overview.fields``).
- ``atl06_windows/`` — ``morton-hive/2`` (explicit windows 2019/2020; 2020
  covers one shard fewer), ``output.pyramid: {orders: [4], all_time: true}``
  → per-window overviews (``2019.zarr``/``2020.zarr``) plus the all-time
  ``all.zarr`` fold at each order-4 ancestor node (D23 naming).

The golden sidecar records, at generation time, the expected node roster per
product (source + overview cell orders), each order's variables, the
per-object ``role``/``zagg_overview`` attrs, and per-field fold checks
(count totals, h_min minima) read back from the written bytes — so the
moczarr reader tests pin against what zagg actually wrote, without zagg
installed. Generation asserts source↔overview count-total parity so a broken
fold can never be enshrined as a golden.

**Reproducibility.** At a fixed zagg sha this generator is byte-reproducible
except for wall clock: the only fields that differ between two runs are
``generated_at`` / ``written_at`` and ``zagg_overview.generation.
max_leaf_timestamp`` (``content_hash`` is stable). So "regenerate and diff" is
a usable drift check, and the reader tests deliberately do **not** value-pin
those fields — they require them, and compare everything else (see
``_without_wallclock`` in ``tests/test_pyramid.py``). Timestamps recorded in
the golden are informational: pinning them against their own recorded copy
would be a self-echo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

#: SERC-area sample points (lat, lon) — a coarse grid over the ~1.5 deg box.
SERC = (38.8901, -76.5600)
#: Per-shard occupied-cell counts (cycled), as in generate_serc_fixture.py.
N_OCCUPIED = {0: 16, 1: 5, 2: 1}

WINDOWS = {
    "2019": ("2019-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00"),
    "2020": ("2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00"),
}
EPOCH = "2018-01-01T00:00:00+00:00"


def serc_shards(order: int = 6) -> list[int]:
    from mortie import geo2mort

    lats = np.linspace(SERC[0] - 0.75, SERC[0] + 0.75, 7)
    lons = np.linspace(SERC[1] - 0.75, SERC[1] + 0.75, 7)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    words = geo2mort(grid_lat.ravel(), grid_lon.ravel(), order=order)
    return sorted(int(w) for w in np.unique(words))


def _zagg_sha() -> str:
    import zagg

    repo = Path(zagg.__file__).resolve().parents[2]
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _seconds(iso: str) -> float:
    """Dataset-unit window bound: seconds since the declared epoch."""
    from zagg.windows import parse_utc

    return (parse_utc(iso) - parse_utc(EPOCH)).total_seconds()


def _write_product(root: Path, cfg, *, shard_windows: dict[int, list[str | None]]) -> dict:
    """Write one product tree with zagg's real writer + overview sweep.

    ``shard_windows`` maps each stamped shard word to the windows to write
    (``[None]`` for an unwindowed product). Returns the read-back manifest.
    """
    import zagg.processing as processing
    from zagg import hive
    from zagg.config import get_agg_fields, get_data_vars, get_output_signature, get_windowing
    from zagg.grids import HealpixGrid
    from zagg.grids.morton import morton_decimal
    from zagg.sweep_overview import sweep_overviews

    grid = HealpixGrid(parent_order=6, child_order=8, layout="fullsphere", config=cfg)
    windowing = get_windowing(cfg)

    def carrier(shard: int, window: str | None) -> tuple[pd.DataFrame, np.ndarray]:
        coords = grid.chunk_coords(shard)
        n = len(coords["morton"])
        # Stable across processes (str hash() is salted): window -> small int.
        salt = 0 if window is None else int(window)
        rng = np.random.default_rng((shard + salt * 7919) % 2**32)
        shards = sorted(shard_windows)
        n_occ = N_OCCUPIED[shards.index(shard) % len(N_OCCUPIED)]
        occupied_rows = np.sort(rng.choice(n, size=min(n_occ, n), replace=False))
        mask = np.zeros(n, dtype=bool)
        mask[occupied_rows] = True
        agg = get_agg_fields(cfg)
        df = pd.DataFrame()
        for var in get_data_vars(cfg):
            if get_output_signature(agg[var])["kind"] == "ragged":
                continue
            if var == "count":
                df[var] = np.where(mask, rng.integers(1, 40, n), 0).astype(np.int32)
            else:
                df[var] = np.where(mask, rng.normal(30.0, 5.0, n), np.nan).astype(np.float32)
        for name, vals in coords.items():
            df[name] = vals
        return df, np.asarray(coords["morton"], dtype=np.uint64)[mask]

    current: dict = {}

    def fake_process_shard(g, shard_key, urls, **kwargs):
        df, occupied = carrier(int(shard_key), current["window"])
        kwargs["write_chunk"](grid.block_index(int(shard_key)), df, {})
        if kwargs.get("occupied_out") is not None:
            kwargs["occupied_out"].append(occupied)
        return pd.DataFrame(), {
            "shard_key": int(shard_key),
            "cells_with_data": int(occupied.size),
            "total_obs": int(occupied.size) * 3,
            "granule_count": 2,
            "files_processed": 2,
            "duration_s": 0.0,
            "error": None,
        }

    root.mkdir(parents=True)
    store_root = str(root)
    hive.ensure_manifest(
        store_root,
        hive.build_manifest(
            grid,
            dataset={"short_name": "ATL06", "version": "007"},
            windowing=windowing,
        ),
    )
    original = processing.process_shard
    processing.process_shard = fake_process_shard
    try:
        for shard, windows in shard_windows.items():
            for label in windows:
                current["window"] = label
                window = None
                if label is not None:
                    start, end = WINDOWS[label]
                    window = {"label": label, "start": _seconds(start), "end": _seconds(end)}
                meta = hive.process_and_write_hive(
                    shard,
                    ["s3://fixture/a.h5", "s3://fixture/b.h5"],
                    grid,
                    {},
                    store_root,
                    cfg,
                    store_kwargs={},
                    window=window,
                )
                assert not meta.get("error"), meta
    finally:
        processing.process_shard = original

    hive.write_root_coverage(
        store_root, hive.build_root_coverage(sorted(shard_windows), 6, source="dispatcher")
    )
    manifest = hive.read_manifest(store_root)
    by_shard = {
        morton_decimal(shard): {w for w in windows} for shard, windows in shard_windows.items()
    }
    counts = sweep_overviews(store_root, manifest, by_shard, store_kwargs={})
    assert counts["written"] and not counts["failed"], counts
    print(f"  sweep[{root.name}]: {counts}")
    return hive.read_manifest(store_root)  # re-read: the sweep adds materialized


def _node_rel(decimal: str) -> str:
    from zagg.sweep import _node_rel as rel

    return rel(decimal)


def _golden_overviews(root: Path, manifest: dict, shard_windows: dict) -> dict:
    """Read back every overview zarr and record the expected reader view."""
    import zarr
    from zagg.grids.morton import morton_decimal

    cell_order = int(manifest["cell_order"])
    shard_order = int(manifest["shard_order"])
    decl = manifest["pyramid"]["overview"]
    windowed = manifest.get("temporal") is not None
    decimals = sorted(morton_decimal(s) for s in shard_windows)
    labels = sorted({w for ws in shard_windows.values() for w in ws if w is not None})
    keys = labels + ["all"] if windowed else [None]
    nodes: dict = {}
    for k in decl["orders"]:
        target = cell_order - (shard_order - k)
        ancestors = sorted({d[: len(d) - (shard_order - k)] for d in decimals})
        for key in keys:
            basename = f"{key}.zarr" if key is not None else "all.zarr"
            objects = []
            variables: set = set()
            totals: dict[str, float] = {}
            minima: list[float] = []
            n_cells = 0
            for node in ancestors:
                path = root / _node_rel(node) / basename
                if not path.exists():
                    continue
                group = zarr.open_group(str(path), mode="r", zarr_format=3)
                attrs = dict(group.attrs)
                assert attrs.get("role") == "overview", (path, attrs.get("role"))
                assert "morton_hive_commit" in attrs, path
                inner = zarr.open_group(str(path / str(target)), mode="r", zarr_format=3)
                arrays = sorted(set(inner.array_keys()) - {"morton"})
                variables |= set(arrays)
                n_cells += int(inner["morton"].shape[0])
                counts = np.asarray(inner["count"][:])
                totals["count"] = totals.get("count", 0) + int(counts.sum())
                h_min = np.asarray(inner["h_min"][:], dtype=np.float64)
                if np.any(~np.isnan(h_min)):
                    minima.append(float(np.nanmin(h_min)))
                objects.append(
                    {
                        "node": node,
                        "window": key if windowed else None,
                        "role": "overview",
                        "zagg_overview": attrs["zagg_overview"],
                    }
                )
            if not objects:
                continue
            entry = {
                "cell_order": target,
                "n_cells": n_cells,
                "variables": sorted(variables),
                "count_total": totals["count"],
                "h_min_min": min(minima) if minima else None,
                "objects": objects,
            }
            nodes.setdefault(str(target), {})[key if windowed else "all"] = entry
    return nodes


def _golden_source(root: Path, manifest: dict, shard_windows: dict) -> dict:
    """Source-order truth per window: variables, count totals, object roster."""
    import zarr
    from zagg.grids.morton import morton_decimal
    from zagg.hive import shard_leaf_path

    cell_order = int(manifest["cell_order"])
    windowed = manifest.get("temporal") is not None
    out: dict = {}
    for shard, windows in sorted(shard_windows.items()):
        for label in windows:
            path = Path(shard_leaf_path(str(root), shard, window=label))
            inner = zarr.open_group(str(path / str(cell_order)), mode="r", zarr_format=3)
            key = label if windowed else "all"
            entry = out.setdefault(
                key,
                {
                    "cell_order": cell_order,
                    "n_cells": 0,
                    "variables": sorted(set(inner.array_keys()) - {"morton"}),
                    "count_total": 0,
                    "objects": [],
                },
            )
            entry["n_cells"] += int(inner["morton"].shape[0])
            entry["count_total"] += int(np.asarray(inner["count"][:]).sum())
            entry["objects"].append(
                {"node": morton_decimal(shard), "window": label, "role": "source"}
            )
    return out


def build(out: Path) -> None:
    from zagg.config import default_config

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shards = serc_shards()
    print(f"{len(shards)} SERC shards at order 6")

    golden: dict = {"zagg_sha": _zagg_sha(), "products": {}}

    # -- atl06: unwindowed, overview orders [4, 2] -> cell orders 6 and 4.
    cfg = default_config("atl06")
    output = dict(cfg.output)
    output["store_layout"] = "hive"
    output["grid"] = {**output["grid"], "parent_order": 6, "child_order": 8}
    output["pyramid"] = {"orders": [4, 2]}
    cfg = replace(cfg, output=output)  # re-runs config validation
    shard_windows: dict[int, list[str | None]] = {s: [None] for s in shards}
    manifest = _write_product(out / "atl06", cfg, shard_windows=shard_windows)
    source = _golden_source(out / "atl06", manifest, shard_windows)
    overviews = _golden_overviews(out / "atl06", manifest, shard_windows)
    for orders in overviews.values():  # fold parity: never enshrine a broken sum
        for entry in orders.values():
            assert entry["count_total"] == source["all"]["count_total"], (entry, source)
    golden["products"]["atl06"] = {
        "spec": manifest["spec"],
        "pyramid_orders": manifest["pyramid"]["overview"]["orders"],
        "cell_order": manifest["cell_order"],
        "shard_order": manifest["shard_order"],
        "source": source,
        "overviews": overviews,
    }

    # -- atl06_windows: /2 explicit windows, overview order [4] + all_time.
    cfg = default_config("atl06")
    data_source = dict(cfg.data_source)
    data_source["variables"] = {
        **data_source["variables"],
        "delta_time": "/{group}/land_ice_segments/delta_time",
    }
    output = dict(cfg.output)
    output["store_layout"] = "hive"
    output["grid"] = {**output["grid"], "parent_order": 6, "child_order": 8}
    output["pyramid"] = {"orders": [4], "all_time": True}
    output["windowing"] = {
        "schedule": "explicit",
        "time_field": "delta_time",
        "epoch": EPOCH,
        "windows": [
            {"label": label, "start": start, "end": end} for label, (start, end) in WINDOWS.items()
        ],
    }
    cfg = replace(cfg, data_source=data_source, output=output)
    # 2020 covers one shard fewer than 2019, so the two windows' overviews
    # (and the all-time fold) differ in both values and coverage.
    shard_windows = {s: ["2019", "2020"] for s in shards[:-1]}
    shard_windows[shards[-1]] = ["2019"]
    manifest = _write_product(out / "atl06_windows", cfg, shard_windows=shard_windows)
    source = _golden_source(out / "atl06_windows", manifest, shard_windows)
    overviews = _golden_overviews(out / "atl06_windows", manifest, shard_windows)
    for orders in overviews.values():
        for key, entry in orders.items():
            want = (
                sum(source[label]["count_total"] for label in source)
                if key == "all"
                else source[key]["count_total"]
            )
            assert entry["count_total"] == want, (key, entry["count_total"], want)
    golden["products"]["atl06_windows"] = {
        "spec": manifest["spec"],
        "pyramid_orders": manifest["pyramid"]["overview"]["orders"],
        "cell_order": manifest["cell_order"],
        "shard_order": manifest["shard_order"],
        "windows": sorted(WINDOWS),
        "source": source,
        "overviews": overviews,
    }

    golden_path = out.parent / f"{out.name}.golden.json"
    golden_path.write_text(json.dumps(golden, indent=1) + "\n")
    n_files = sum(1 for f in out.rglob("*") if f.is_file())
    size_kb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024
    print(f"wrote {out}: {n_files} objects, {size_kb:.0f} KB (zagg {golden['zagg_sha']})")
    print(f"wrote {golden_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)

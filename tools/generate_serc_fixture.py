"""Generate the committed SERC fixture store with zagg's REAL writer.

Run from a zagg checkout's environment (zagg is deliberately not a moczarr
dependency)::

    cd ../zagg && uv run python ../moczarr/tools/generate_serc_fixture.py \
        --out ../moczarr/tests/data/serc_hive

Every byte the reader tests consume is produced by zagg's own write path —
``build_manifest``/``ensure_manifest``, ``process_and_write_hive`` (leaf
template, dense chunk write, coverage bitmap sidecar, commit stamp) and
``write_root_coverage`` — so writer<->reader drift fails moczarr's suite on
whichever side moved (the ratified fixture posture, espg/moczarr#1; the
long-term home is a public source.coop store once the spec stops drifting).

Layout: the order-6 shards covering a ~1.5 degree box around the NEON SERC
site (38.8901N, -76.5600W), atl06 default aggregation (no ragged fields),
parent order 6 / child order 8 (16 cells per shard — deliberately tiny).
Observations are synthetic and deterministic: each shard occupies a
different subset of its cells, values seeded from the shard word. One extra
leaf is written as DEBRIS (template, no stamp) to pin the skip posture.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

#: SERC-area sample points (lat, lon) — a coarse grid over the ~1.5 deg box.
SERC = (38.8901, -76.5600)
#: Shard deliberately written UNSTAMPED (debris). Chosen from the covered set.
N_OCCUPIED = {0: 16, 1: 5, 2: 1}  # per-shard occupied-cell counts (cycled)


def serc_shards(order: int = 6) -> list[int]:
    from mortie import geo2mort

    lats = np.linspace(SERC[0] - 0.75, SERC[0] + 0.75, 7)
    lons = np.linspace(SERC[1] - 0.75, SERC[1] + 0.75, 7)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    words = geo2mort(grid_lat.ravel(), grid_lon.ravel(), order=order)
    return sorted(int(w) for w in np.unique(words))


def build(out: Path) -> None:
    import zagg.processing as processing
    from zagg import hive
    from zagg.config import (
        default_config,
        get_agg_fields,
        get_data_vars,
        get_output_signature,
    )
    from zagg.grids import HealpixGrid

    cfg = default_config("atl06")
    cfg.output["store_layout"] = "hive"
    grid = HealpixGrid(parent_order=6, child_order=8, layout="fullsphere", config=cfg)

    shards = serc_shards()
    debris, stamped = shards[-1], shards[:-1]
    print(f"{len(stamped)} stamped shards + 1 debris: {debris}")

    def carrier(shard: int) -> tuple[pd.DataFrame, np.ndarray]:
        coords = grid.chunk_coords(shard)
        n = len(coords["morton"])
        rng = np.random.default_rng(shard % 2**32)
        n_occ = N_OCCUPIED[stamped.index(shard) % len(N_OCCUPIED)]
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

    def fake_process_shard(g, shard_key, urls, **kwargs):
        df, occupied = carrier(int(shard_key))
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

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = str(out)

    hive.ensure_manifest(
        root, hive.build_manifest(grid, dataset={"short_name": "ATL06", "version": "007"})
    )
    original = processing.process_shard
    processing.process_shard = fake_process_shard
    try:
        for shard in stamped:
            meta = hive.process_and_write_hive(
                shard,
                ["s3://fixture/a.h5", "s3://fixture/b.h5"],
                grid,
                {},
                root,
                cfg,
                store_kwargs={},
            )
            assert not meta.get("error"), meta
    finally:
        processing.process_shard = original

    # Debris: the leaf template exists (the torn-worker shape) but no stamp.
    from zagg.store import open_store

    grid.emit_shard_template(open_store(hive.shard_leaf_path(root, debris)), overwrite=True)

    # Root MOC lists the STAMPED shards only (debris is not coverage).
    hive.write_root_coverage(root, hive.build_root_coverage(stamped, 6, source="dispatcher"))

    n_files = sum(1 for _ in out.rglob("*") if _.is_file())
    size_kb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024
    print(f"wrote {root}: {n_files} objects, {size_kb:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)

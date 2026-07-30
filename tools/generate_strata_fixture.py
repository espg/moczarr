"""Generate the committed strata fixture + parity goldens with zagg's REAL writer.

Run from a zagg checkout's environment (zagg is deliberately not a moczarr
dependency)::

    cd ../zagg && uv run python ../moczarr/tools/generate_strata_fixture.py \
        --out ../moczarr/tests/data

The ratified fixture posture (``tools/generate_serc_fixture.py``): every byte
the reader tests consume is produced by zagg's own write path —
``hive.ensure_manifest`` + ``hive.process_and_write_hive`` (leaf template,
sharded dense + ragged writes, coverage bitmap sidecar, commit stamp) — so
writer<->reader drift fails moczarr's suite on whichever side moved. ONE
fixture explicitly designed to serve issues #19, #20, and #21: a tiny
SERC-area hive leaf with located ``h_tdigest_signal``/``h_tdigest_noise``
strata (payloads + ``{field}_locations`` siblings, ``stratum``/
``signal_threshold`` provenance attrs), the packed ``composition`` word, and
``count``.

Geometry: the order-4 shard over the NEON SERC site (38.8901N, -76.5600W),
chunk order 5 / cell order 6 — 16 cells, K = 4 inner chunks of 4 cells,
sharded (the hive default). Chunk ordinal 2 is left EMPTY (the shard-index
absence sentinel); populated chunks carry empty cells (the ``b""`` fill);
noise-only cells pin the empty ``(0, 2)`` signal payload and the 3-state
mask; the 300-obs cell merges centroids (weight > 1) whose location words
are common ancestors.

Alongside the store it writes:

- ``strata_hive.expected.json`` — decode-level truth computed from the
  generator's INPUTS (never read back through a zagg reader), plus the O11
  content hashes: centroids, location words and composition words as decimal
  strings, per-stratum exact counts.
- ``strata_goldens/*.npy`` — tensor-level truth computed by zagg's
  ``readers.tdigest_tensor.read_tensors`` at generation time (issue #19
  parity pin: expected tensors/masks/windows/morton ids for the signal and
  noise fields per read chunk, and the signal field assembled at
  ``block_order`` = the shard order).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

#: NEON SERC site — the fixture shard is the order-4 cell over it.
SERC = (38.8901, -76.5600)
#: Chunk ordinal (0..3) deliberately left empty — the absence-sentinel pin.
EMPTY_CHUNK = 2
#: t-digest compression budget: small enough that the 300-obs cell merges
#: observations into weight>1 centroids (common-ancestor location words).
DELTA = 16
#: The composition golden-word photon: per-surface confidences at threshold
#: 2 pack lanes [255, 0, 0, 255, 0, 0, 0, 255] = 0xFF000000FF0000FF.
GOLDEN_CONF = (4, -1, 0, 3, 1)
#: Tensor-golden reader parameters (read_tensors defaults + the block order).
GOLDEN_PARAMS = {
    "n_bins": 128,
    "resolution": 0.5,
    "bottom": 0.05,
    "top": 0.95,
    "dtype": "uint32",
    "block_order": 4,
}
#: (chunk ordinal, local cell, n obs, kind). Chunk 3 pairs a mixed cell with
#: a noise-only cell so the per-chunk signal mask carries a genuine ``1``.
PLAN = [
    (0, 0, 40, "mixed"),
    (0, 2, 1, "golden"),
    (1, 1, 5, "noise_only"),
    (3, 1, 7, "noise_only"),
    (3, 3, 300, "mixed"),
]


def _cell_photons(rng, n, *, signal_frac=0.6):
    """Synthetic photons: heights + 5 confidence columns, ``n`` rows."""
    h = np.round(rng.normal(30.0, 5.0, n), 3).astype(np.float64)
    conf = np.full((n, 5), -1, dtype=np.int64)
    n_sig = int(round(n * signal_frac))
    for i in range(n):
        if i < n_sig:  # signal: strongest confidence 2..4 on 1-2 surfaces
            surfaces = rng.choice(5, size=int(rng.integers(1, 3)), replace=False)
            conf[i, surfaces] = rng.integers(2, 5)
        else:  # noise: everything below threshold
            conf[i] = rng.integers(-2, 2, 5)
    return h, conf


def _point_words(grid, cell_word, n, rng):
    """Order-29 point-kind words jittered around the cell's center."""
    from mortie import MortonIndexArray
    from zagg.grids.morton import morton_words

    lat, lon = grid.cell_centers(np.array([cell_word], dtype=np.uint64))
    lat0, lon0 = float(np.asarray(lat).ravel()[0]), float(np.asarray(lon).ravel()[0])
    lats = np.clip(lat0 + rng.uniform(-1e-5, 1e-5, n), -89.9, 89.9)
    lons = lon0 + rng.uniform(-1e-5, 1e-5, n)
    return morton_words(MortonIndexArray.from_latlon(lats, lons, points=True))


def _config():
    from zagg.config import PipelineConfig

    variables: dict = {
        "count": {"function": "len", "source": "h", "dtype": "int32", "fill_value": 0}
    }
    for stratum in ("signal", "noise"):
        variables[f"h_tdigest_{stratum}"] = {
            "kind": "ragged",
            "function": "zagg.stats.tdigest.build_tdigest_where",
            "source": "h",
            "location": "leaf_id",
            "inner_shape": [2],
            "dtype": "float32",
            "fill_value": 0,
            "params": {"delta": DELTA},
            "attrs": {"stratum": stratum, "signal_threshold": 2},
        }
    variables["composition"] = {
        "function": "zagg.stats.composition.pack_composition",
        "source": "h",
        "dtype": "uint64",
        "fill_value": 0,
        "params": {"threshold": 2},
        "attrs": {
            "composition": {
                "spec": "zagg-composition/1",
                "lanes": [
                    "land",
                    "ocean",
                    "sea_ice",
                    "land_ice",
                    "inland_water",
                    "low",
                    "med",
                    "high",
                ],
                "of": "h_tdigest_signal",
                "threshold": 2,
            }
        },
    }
    return PipelineConfig(
        data_source={"groups": ["g"]},
        aggregation={
            "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
            "variables": variables,
        },
        output={
            "store_layout": "hive",
            "grid": {
                "type": "healpix",
                "parent_order": 4,
                "child_order": 6,
                "chunk_inner": 5,
                "sharded": True,
            },
        },
    )


def _build_cells(grid, shard):
    """Per-cell synthetic inputs and their expected decoded values."""
    from zagg.stats.composition import pack_composition
    from zagg.stats.tdigest import build_tdigest_where

    rng = np.random.default_rng(19)
    children = grid.children(shard)
    by_chunk: dict = {}
    expected = []
    for chunk, local, n, kind in PLAN:
        cell_index = chunk * grid.cells_per_chunk + local
        cell_word = int(children[cell_index])
        h, conf = _cell_photons(rng, n, signal_frac={"mixed": 0.6}.get(kind, 0.0))
        if kind == "golden":
            conf = np.array([GOLDEN_CONF], dtype=np.int64)
        if kind == "noise_only":
            conf[:] = rng.integers(-2, 2, conf.shape)
        record: dict = {"index": cell_index, "morton": str(cell_word), "count": n}
        fields: dict = {"count": n}
        words = _point_words(grid, cell_word, n, rng)
        signal = (conf >= 2).any(axis=1)
        for stratum, mask in (("signal", signal), ("noise", ~signal)):
            digest, locs = build_tdigest_where(h, DELTA, where=mask, locations=words)
            fields[f"h_tdigest_{stratum}"] = (digest, locs)
            record[f"h_tdigest_{stratum}"] = [[float(m), float(w)] for m, w in digest]
            record[f"h_tdigest_{stratum}_locations"] = [str(int(w)) for w in locs]
        word = pack_composition(
            h,
            conf_land=conf[:, 0],
            conf_ocean=conf[:, 1],
            conf_sea_ice=conf[:, 2],
            conf_land_ice=conf[:, 3],
            conf_inland_water=conf[:, 4],
            threshold=2,
        )
        fields["composition"] = word
        record["composition"] = str(word)
        record["n_signal"] = int(signal.sum())
        by_chunk.setdefault(chunk, {})[local] = fields
        expected.append(record)
    return by_chunk, expected


def _fake_process_shard(grid, by_chunk):
    """A ``process_shard`` stand-in feeding the REAL sharded leaf write path."""

    def fake(g, shard_key, urls, **kwargs):
        occupied = []
        for ordinal, (block, children) in enumerate(grid.iter_chunks(int(shard_key))):
            cells = by_chunk.get(ordinal, {})
            if not cells:
                kwargs["chunk_results"].append((block, pd.DataFrame(), {}))
                continue
            n = grid.cells_per_chunk
            df = pd.DataFrame({"morton": np.asarray(children, dtype=np.uint64)})
            df["count"] = np.array(
                [cells.get(i, {}).get("count", 0) for i in range(n)], dtype=np.int32
            )
            df["composition"] = np.array(
                [cells.get(i, {}).get("composition", 0) for i in range(n)],
                dtype=np.uint64,
            )
            ragged: dict = {}
            for field in ("h_tdigest_signal", "h_tdigest_noise"):
                ids = sorted(i for i, f in cells.items() if field in f)
                ragged[field] = (
                    [cells[i][field][0] for i in ids],
                    ids,
                    [cells[i][field][1] for i in ids],
                )
            kwargs["chunk_results"].append((block, df, ragged))
            occupied.extend(int(children[i]) for i in sorted(cells))
        kwargs["occupied_out"].append(np.asarray(occupied, dtype=np.uint64))
        return pd.DataFrame(), {
            "shard_key": int(shard_key),
            "cells_with_data": len(occupied),
            "total_obs": sum(n for _c, _l, n, _k in PLAN),
            "granule_count": 1,
            "files_processed": 1,
            "duration_s": 0.0,
            "error": None,
        }

    return fake


def _o11_hashes(leaf_path: str) -> dict:
    """The spec §5 O11 recipe over every array beneath the leaf root."""
    import zarr

    group = zarr.open_group(zarr.storage.LocalStore(leaf_path), mode="r", zarr_format=3)
    hashes: dict[str, str] = {}
    for key, node in group.members(max_depth=None):
        if not isinstance(node, zarr.Array):
            continue
        values = np.ascontiguousarray(node[...])
        if values.dtype.kind == "O":  # vlen: length-prefixed payloads, C order
            digest = hashlib.sha256()
            for element in values.ravel(order="C"):
                payload = b"" if element is None else bytes(element)
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
            hashes[key] = digest.hexdigest()
            continue
        if values.dtype.byteorder == ">":
            values = values.astype(values.dtype.newbyteorder("<"))
        hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
    combined = hashlib.sha256("\n".join(sorted(hashes.values())).encode()).hexdigest()
    return {"arrays": hashes, "combined": combined}


def _tensor_goldens(leaf: Path, group: str, goldens_dir: Path) -> dict:
    """Tensor truth via zagg's shipping reader; committed as ``.npy`` files."""
    import zarr
    from zagg.readers.tdigest_tensor import has_exact_occupancy, read_tensors

    store = zarr.storage.LocalStore(str(leaf))
    assert has_exact_occupancy(store), "leaf must carry exact coverage.moc occupancy"
    if goldens_dir.exists():
        shutil.rmtree(goldens_dir)
    goldens_dir.mkdir(parents=True)
    manifest: dict = {"params": dict(GOLDEN_PARAMS)}

    for stratum in ("signal", "noise"):
        blocks = list(read_tensors(store, f"{group}/h_tdigest_{stratum}"))
        np.save(goldens_dir / f"{stratum}_tensors.npy", np.stack([b[0] for b in blocks]))
        np.save(goldens_dir / f"{stratum}_masks.npy", np.stack([b[1] for b in blocks]))
        np.save(
            goldens_dir / f"{stratum}_windows.npy",
            np.array([b[2] for b in blocks], dtype=np.float64),
        )
        np.save(
            goldens_dir / f"{stratum}_morton.npy",
            np.array([b[3] for b in blocks], dtype=np.uint64),
        )
        manifest[stratum] = {"n_blocks": len(blocks)}

    (block,) = list(
        read_tensors(
            store,
            f"{group}/h_tdigest_signal",
            block_order=int(GOLDEN_PARAMS["block_order"]),
        )
    )
    np.save(goldens_dir / "signal_block_tensor.npy", block[0])
    np.save(goldens_dir / "signal_block_mask.npy", block[1])
    np.save(goldens_dir / "signal_block_window.npy", np.array(block[2], dtype=np.float64))
    np.save(goldens_dir / "signal_block_morton.npy", np.array(block[3], dtype=np.uint64))
    return manifest


def build(out: Path) -> None:
    import zagg
    import zagg.processing as processing
    from mortie import geo2mort
    from zagg import hive
    from zagg.grids import HealpixGrid
    from zagg.grids.morton import morton_decimal

    cfg = _config()
    grid = HealpixGrid(4, 6, layout="fullsphere", config=cfg, chunk_inner=5, sharded=True)
    shard = int(geo2mort(np.array([SERC[0]]), np.array([SERC[1]]), order=4)[0])
    shard_key = morton_decimal(shard)
    by_chunk, expected_cells = _build_cells(grid, shard)
    assert EMPTY_CHUNK not in by_chunk

    store_dir = out / "strata_hive"
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True)
    root = str(store_dir)
    hive.ensure_manifest(
        root,
        hive.build_manifest(grid, dataset={"short_name": "STRATA_FIXTURE", "version": "1"}),
    )
    original = processing.process_shard
    processing.process_shard = _fake_process_shard(grid, by_chunk)
    try:
        meta = hive.process_and_write_hive(
            shard,
            ["s3://fixture/a.h5"],
            grid,
            {},
            root,
            cfg,
            store_kwargs={},
        )
    finally:
        processing.process_shard = original
    assert meta.get("error") is None, meta

    leaf_rel = hive.shard_leaf_path("", shard).lstrip("/")
    goldens = _tensor_goldens(store_dir / leaf_rel, grid.group_path, out / "strata_goldens")
    expected = {
        "shard": shard_key,
        "leaf": leaf_rel,
        "group": grid.group_path,
        "shard_order": 4,
        "chunk_order": 5,
        "cell_order": 6,
        "cells_per_chunk": grid.cells_per_chunk,
        "chunks_per_shard": grid.chunks_per_shard,
        "empty_chunk": EMPTY_CHUNK,
        "delta": DELTA,
        "signal_threshold": 2,
        "generator": {"zagg_version": zagg.__version__, "seed": 19},
        "goldens": goldens,
        "cells": expected_cells,
        "content_hashes": _o11_hashes(str(store_dir / leaf_rel)),
    }
    (out / "strata_hive.expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"strata_hive: shard {shard_key}, leaf {leaf_rel}, {len(expected_cells)} cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "data",
    )
    build(parser.parse_args().out)

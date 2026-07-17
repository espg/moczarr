"""``open_hive``: a morton-hive store as one lazy xarray Dataset.

The §5 reader flow, arithmetic-first (D10): manifest GET → root coverage
MOC ∩ AOI → hive paths by string arithmetic → stamped-leaf opens → concat
along the cell dimension. The discovery walk runs only when the root MOC is
absent/unusable (D9: caches degrade to the walk, never to wrong answers).
Debris (unstamped leaves) is skipped silently — absence of a stamp IS the
answer (D4).

AOI semantics: ``aoi`` is a morton cover — packed ``uint64`` words or
decimal strings, mixed orders allowed. Shards are rejected arithmetically
(root MOC ∩ AOI), then per leaf by the tier-0 box off the stamp already
fetched, and finally rows are subset exactly on the ``morton`` coordinate
(tier 2 — the coordinate is the truth; the MOC tiers are only indexes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from moczarr.convention import (
    HIVE_SPEC_V2,
    leaf_path,
    morton_word,
    split_leaf_name,
    validate_label,
)
from moczarr.coverage import (
    aoi_mask,
    box_and,
    parse_leaf_coverage,
    ranges_words,
    root_coverage_and,
)
from moczarr.fabricate import fabricate_cell_ids as _fabricate_cell_ids
from moczarr.store import (
    load_root_coverage,
    open_object_store,
    read_commits,
    read_manifest,
    walk_leaves,
)


def _aoi_words(aoi) -> np.ndarray:
    """Normalize an AOI cover to packed ``uint64`` words (strings accepted)."""
    values = list(np.asarray(aoi).ravel()) if np.asarray(aoi).ndim else [aoi]
    return np.asarray([morton_word(v) for v in values], dtype=np.uint64)


def _candidate_leaves(
    store_root: str,
    manifest: dict,
    aoi,
    window: str | None,
    *,
    store: Any = None,
    concurrency: int | None = None,
) -> list[str]:
    """Store-relative leaf paths to try, ascending in packed-word order.

    Arithmetic (root MOC) when possible; the walk otherwise. A windowed
    (``morton-hive/2``) store needs an explicit ``window`` for the
    arithmetic path — with ``window=None`` the walk enumerates what exists
    and the error message lists the labels.
    """
    windowed = manifest["spec"] == HIVE_SPEC_V2
    if window is not None:
        validate_label(window)
        if not windowed:
            raise ValueError(
                f"window={window!r} on a {manifest['spec']} store: unwindowed stores "
                f"have no window leaves (schedule: none)"
            )
    envelope = load_root_coverage(store_root, store=store)
    if envelope is not None and not (windowed and window is None):
        words = ranges_words(envelope) if aoi is None else root_coverage_and(envelope, aoi)
        if aoi is not None and words.size:
            # The MOC intersection keeps the FINER element of each overlapping
            # pair (a cell-order AOI member intersected with its shard yields
            # the cell), so coarsen back to shard order to name the leaves.
            from mortie import clip2order

            words = np.unique(clip2order(int(manifest["shard_order"]), words))
        return [leaf_path(int(w), window=window) for w in np.sort(words)]
    # Walk fallback (no usable root MOC), and the windowed-discovery case.
    found: dict[str, int] = {}
    labels: set[str] = set()
    for rel in walk_leaves(store_root, store=store, concurrency=concurrency):
        shard, label = split_leaf_name(rel.rsplit("/", 1)[-1])
        labels.add(label if label is not None else "<none>")
        if label != window:
            continue
        found[rel] = morton_word(shard)
    if windowed and window is None and labels - {"<none>"}:
        raise ValueError(
            f"{store_root} is a windowed ({HIVE_SPEC_V2}) store; pass window=... "
            f"(labels present: {sorted(labels - {'<none>'})})"
        )
    if aoi is not None:
        from mortie import moc_and

        found = {
            rel: w
            for rel, w in found.items()
            if moc_and(np.asarray([w], dtype=np.uint64), aoi).size
        }
    return sorted(found, key=lambda rel: found[rel])


def open_hive(
    store_root: str,
    *,
    aoi=None,
    window: str | None = None,
    anonymous: bool = False,
    fabricate_cell_ids: bool | str = "auto",
    decode: bool = False,
    concurrency: int | None = 32,
    xr_kwargs: dict[str, Any] | None = None,
    **store_kwargs: Any,
):
    """Open a morton-hive store as one xarray Dataset.

    Parameters
    ----------
    store_root : str
        Store root (local directory or ``s3://bucket/prefix``).
    aoi : array-like, optional
        Morton cover of the area of interest — packed ``uint64`` words or
        decimal strings, mixed orders allowed. Shards and rows outside the
        cover are excluded (rows exactly, via the ``morton`` coordinate).
    window : str, optional
        Window label for a time-windowed (``morton-hive/2``) store. Omitted
        on such a store, the error lists the labels that exist.
    anonymous : bool, optional
        Unsigned S3 requests (public buckets).
    fabricate_cell_ids : {"auto", True, False}, optional
        NESTED ``cell_ids`` posture (englacial/zagg#262: "NESTED is
        fabricated, never stored"). ``"auto"`` (default): a stored
        ``cell_ids`` coordinate is kept untouched; when absent (morton-only
        store) an exact NESTED view is fabricated from the ``morton``
        coordinate via :func:`moczarr.fabricate.fabricate_cell_ids`.
        ``True`` always fabricates (replacing any stored array — exact, so
        a no-op on dual-written stores); ``False`` never fabricates.
        Fabrication runs once post-concat on the final ``morton``
        coordinate — equivalent to per-leaf (the same words, and
        ``mort2healpix`` is elementwise) but a single vectorized call.
        The fabricated ``cell_ids`` is a Python-side convenience view: the
        dataset-level ``attrs["dggs"]`` block is left untouched, so on a
        morton-only store it still advertises the morton scheme while the
        added coordinate is NESTED. Re-serializing such a result is not
        internally consistent; the authoritative morton-only ``dggs``
        discriminator is owned by the zagg#262 convention work.
    decode : bool, optional
        Assign the xdggs ``MortonIndex`` to the ``morton`` coordinate before
        returning (``moczarr.dggs.decode``), enabling the ``ds.dggs``
        accessor. Requires the ``moczarr[xdggs]`` extra; the default leaves
        the result index-free and xdggs-free.
    concurrency : int or None, optional
        Maximum in-flight metadata requests (the candidate leaves' stamp
        GETs, and the discovery walk's per-level LISTs), default 32 — the
        zarr-python knob vocabulary. ``None`` or ``1`` runs the serial path
        (debugging). Leaf DATA opens stay serial either way (issue #5,
        measured in the phase-3 bench; revisit with the lazy-index work).
    xr_kwargs : dict, optional
        Extra keyword arguments for each leaf's ``xarray.open_zarr`` (e.g.
        ``chunks={}`` for dask-backed laziness).
    **store_kwargs
        Extra keyword arguments for the object store (``region=...`` etc.).

    Returns
    -------
    xarray.Dataset
        Leaves concatenated along the cell dimension in ascending packed
        morton order, with ``morton``/``cell_ids`` as coordinates and the
        manifest summary under ``attrs["morton_hive"]``. Raises
        ``ValueError`` when the root is not a hive store or nothing
        intersects the query.
    """
    import xarray as xr
    from zarr.storage import ObjectStore

    if fabricate_cell_ids not in ("auto", True, False):
        raise ValueError(
            f"fabricate_cell_ids={fabricate_cell_ids!r}: expected 'auto', True, or False"
        )
    if fabricate_cell_ids != "auto":
        fabricate_cell_ids = bool(fabricate_cell_ids)
    if anonymous:
        store_kwargs.setdefault("anonymous", True)
    # ONE store construction pair for the whole open (issue #5): the obstore
    # handle serves every JSON/sidecar read; the zarr wrapper serves every
    # leaf open via deep paths through the parentless digit tree.
    obstore_store = open_object_store(store_root, **store_kwargs)
    zarr_store = ObjectStore(obstore_store, read_only=True)
    manifest = read_manifest(store_root, store=obstore_store)
    if manifest is None:
        raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    aoi_words = _aoi_words(aoi) if aoi is not None else None
    group = str(manifest["cell_order"])
    opened = []
    candidates = _candidate_leaves(
        store_root, manifest, aoi_words, window, store=obstore_store, concurrency=concurrency
    )
    stamps = read_commits(store_root, candidates, store=obstore_store, concurrency=concurrency)
    for rel, stamp in zip(candidates, stamps):
        if stamp is None:
            continue  # debris or a MOC-listed shard whose leaf is gone (D4)
        if aoi_words is not None:
            coverage = parse_leaf_coverage(stamp)
            if coverage is not None and coverage.get("box"):
                if box_and(coverage, aoi_words).size == 0:
                    continue  # conservative reject: false positives only
        ds = xr.open_zarr(
            zarr_store,
            group=f"{rel}/{group}",
            consolidated=False,
            zarr_format=3,
            **(xr_kwargs or {}),
        )
        coords = [name for name in ("morton", "cell_ids") if name in ds]
        ds = ds.set_coords(coords)
        if aoi_words is not None and "morton" in ds.coords:
            keep = aoi_mask(np.asarray(ds["morton"].values, dtype=np.uint64), aoi_words)
            if not keep.any():
                continue
            ds = ds.isel({ds["morton"].dims[0]: keep})
        opened.append(ds)
    if not opened:
        raise ValueError(
            f"nothing to open at {store_root}"
            + (" for the given AOI" if aoi_words is not None else "")
            + (f" in window {window!r}" if window is not None else "")
        )
    dim = opened[0]["morton"].dims[0] if "morton" in opened[0].coords else "cells"
    result = xr.concat(opened, dim=dim) if len(opened) > 1 else opened[0]
    if "morton" in result.coords and (
        fabricate_cell_ids is True
        or (fabricate_cell_ids == "auto" and "cell_ids" not in result.coords)
    ):
        ids = _fabricate_cell_ids(
            np.asarray(result["morton"].values, dtype=np.uint64),
            level=int(manifest["cell_order"]),
            # +1 frame vs a direct call so the >24 warning lands on the
            # user's open_hive(...) line, not this internal call site.
            _stacklevel=4,
        )
        result = result.assign_coords(cell_ids=(result["morton"].dims, ids))
    result.attrs["morton_hive"] = {
        k: manifest[k] for k in ("spec", "cell_order", "shard_order", "dataset")
    }
    if decode:
        from moczarr import dggs  # lazy: raises the pointed extra hint when absent

        result = dggs.decode(result)
    return result

"""HHDC tensor profile: per-cell t-digests → ``(side, side, n_bins)`` tensors.

The first profile over the generic decode layer (:mod:`moczarr.ragged`,
issue #19): :func:`read_tensors` yields ``(tensor, mask, (offset, gain),
morton_index)`` per coverage block — the reader contract ratified on
englacial/zagg#336/#339, reproduced here **bit-identically** (pinned by the
committed ``tests/data/strata_goldens`` and by live parity against zagg's
``readers.tdigest_tensor.read_tensors`` when the ``zagg`` extra is
installed).

Three deliberate seams:

- **Layout kernel** — a cell's 2-D position is the bit deinterleave of its
  chunk-local nested rank (mortie spec §8, ``rank_to_xy``/``xy_to_rank``,
  frozen for mortie 1.x), never a row-major reshape: nested order is a
  Z-order curve, so ``divmod(rank, side)`` would scramble the block
  spatially. Orientation is pinned once in :func:`rank_to_rowcol` — row =
  ``y``, col = ``x``, ``tensor[0, 0]`` at the block subtree's **south
  corner** (gridlook's texture convention, ``bit_combine(j, i)``), matching
  zagg's ``readers/_layout.py`` exactly.
- **Digest algebra** — rasterization needs zagg's t-digest CDF/quantile
  (``cdf_from_tdigest``/``quantile_from_tdigest``). It is IMPORTED from
  zagg, never vendored (vendoring is parity drift by construction): install
  the extra, ``pip install 'moczarr[zagg]'``. The import is lazy, so
  everything else in this module (masks, occupancy, the layout kernel)
  works without it. The seam is the *algebra*, and only that: the reader
  logic around it (:func:`rasterize_cell`, :func:`chunk_z_range`, the
  occupancy/mask helpers, the :func:`read_tensors` body) is a **port** of
  zagg's ``readers/tdigest_tensor.py``, several functions logic-identical.
  That duplication is deliberate and temporary — zagg's reader is expected to
  retire in moczarr's favour — but until then it is a real drift surface,
  held by the committed goldens plus ``TestLiveParity``. The live leg needs
  zagg's post-englacial/zagg#339 reader surface, which no zagg *release*
  carries yet (0.39.0's reader predates the deinterleave), so it is a
  checkout-only leg: against the declared ``zagg>=0.39`` floor it skips, and
  the enforcement that reaches CI is the goldens.
- **Occupancy** — the mask channel decodes the hive leaf's ``coverage.moc``
  occupancy sidecar through moczarr's own frozen bitmap convention
  (:func:`moczarr.coverage.decode_bitmap`), never through zagg.

Mask semantics (the englacial/zagg#334 strata upgrade is data-driven): ``0``
= unobserved, ``1`` = observed but no stored digest on **the field being
read**, ``2`` = observed with a stored digest. ``1`` is symmetric, not a
statement about one stratum: reading a signal field it marks the cells whose
photons were all noise, reading the noise field it marks the signal-only
cells, and a cell that is observed with *both* strata empty reports ``1`` on
both fields. A store without exact
occupancy (no commit stamp — every flat store — or a box-only envelope, or
a missing sidecar) degrades to the 2-state ``{0, 2}`` populated/not mask;
the yielded mask does not say which regime it is in, so consumers keying on
``mask == 1`` MUST check :func:`has_exact_occupancy` first.

Store-scoped like the decode layer: hive products are read one leaf at a
time (open the leaf store, pass the in-leaf field path); leaf discovery
stays with :func:`moczarr.open_hive` / the coverage MOC.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import chain, groupby
from typing import Literal

import numpy as np
import zarr
from mortie import clip2order, rank_to_xy, xy_to_rank
from zarr.abc.store import Store

from moczarr.convention import COMMIT_ATTR, decimal_order, morton_decimal
from moczarr.coverage import decode_bitmap, parse_leaf_coverage
from moczarr.ragged import _morton_words, decode_cell, iter_populated_chunks, open_ragged

__all__ = [
    "chunk_z_range",
    "has_exact_occupancy",
    "rank_to_rowcol",
    "rasterize_cell",
    "read_tensors",
    "rowcol_to_rank",
]

FitMode = Literal["raise", "degrade_resolution", "collapse_bins"]
TensorDtype = Literal["uint16", "uint32", "float32"]

_TENSOR_DTYPES: dict[str, np.dtype] = {
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "float32": np.dtype(np.float32),
}


# --------------------------------------------------------------------------- #
# layout kernel (mortie spec §8; orientation pinned here, once)
# --------------------------------------------------------------------------- #


def rank_to_rowcol(rank, depth: int):
    """``(row, col)`` tensor position of a chunk-local nested rank.

    ``rank`` is the cell's position ``0..4**depth - 1`` within its
    depth-``depth`` subtree (scalar or array; the same index the ragged
    writers use on the cells axis). Returns ``(row, col) = (y, x)``:
    ``x`` gathers the rank's even bits, ``y`` its odd bits (mortie spec §8),
    with ``tensor[0, 0]`` at the subtree's south corner — rows advance
    toward the north-west edge, columns toward the north-east edge (the
    gridlook texture convention; identical to zagg ``readers/_layout``).
    """
    x, y = rank_to_xy(rank, depth)
    return y, x


def rowcol_to_rank(row, col, depth: int):
    """Chunk-local nested rank at a ``(row, col)`` tensor position.

    Inverse of :func:`rank_to_rowcol`: ``row`` is ``y``, ``col`` is ``x``
    (scalars or arrays; values must be ``< 2**depth``).
    """
    return xy_to_rank(col, row, depth)


# --------------------------------------------------------------------------- #
# digest algebra (zagg-owned; the moczarr[zagg] extra)
# --------------------------------------------------------------------------- #


def _tdigest_algebra():
    """zagg's ``(cdf_from_tdigest, quantile_from_tdigest)``, or a pointed error."""
    try:
        from zagg.stats.tdigest import cdf_from_tdigest, quantile_from_tdigest
    except ImportError as exc:
        raise ImportError(
            "HHDC rasterization needs zagg's t-digest algebra "
            "(cdf_from_tdigest/quantile_from_tdigest) — imported, never vendored. "
            "Install the extra: pip install 'moczarr[zagg]'"
        ) from exc
    return cdf_from_tdigest, quantile_from_tdigest


def rasterize_cell(
    digest: np.ndarray,
    z_lo: float,
    resolution: float,
    n_bins: int,
) -> np.ndarray:
    """Rasterize one cell's t-digest into ``n_bins`` per-bin counts.

    Bins are evenly spaced in value-space: bin ``i`` covers ``[z_lo +
    i*resolution, z_lo + (i+1)*resolution)``, and its count is the digest's
    reconstructed weight in that interval (``cdf(edge_{i+1}) -
    cdf(edge_i)``). Weight outside the window is dropped — the window is
    fixed; :func:`chunk_z_range`'s fit policy guards against truncation.
    Returns float64 counts (not yet cast to the output dtype); an empty
    digest yields zeros.
    """
    cdf_from_tdigest, _ = _tdigest_algebra()
    if len(digest) == 0:
        return np.zeros(n_bins, dtype=np.float64)
    edges = z_lo + resolution * np.arange(n_bins + 1, dtype=np.float64)
    cdf = np.asarray(cdf_from_tdigest(digest, edges), dtype=np.float64)
    counts = np.diff(cdf)
    # CDF is monotonic non-decreasing, so counts are ≥ 0 up to float noise.
    np.clip(counts, 0.0, None, out=counts)
    return counts


def _cell_tail_bounds(digest: np.ndarray, bottom: float, top: float) -> tuple[float, float] | None:
    """``(lo, hi)`` = (``bottom``, ``top``) quantiles, or ``None`` if empty."""
    _, quantile_from_tdigest = _tdigest_algebra()
    if len(digest) == 0:
        return None
    lo = quantile_from_tdigest(digest, bottom)
    hi = quantile_from_tdigest(digest, top)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo, hi


def chunk_z_range(
    digests: list[np.ndarray],
    *,
    n_bins: int,
    resolution: float,
    bottom: float,
    top: float,
    fit: FitMode,
) -> tuple[float, int, float]:
    """Derive a block's z-window and apply the fit policy.

    Per cell, the ``bottom``/``top`` quantiles trim the tails; the window
    floor is ``z_lo = floor(min lo_c)`` and the fixed window spans ``n_bins
    * resolution``. When the trimmed range does not fit, ``fit`` decides:

    - ``"raise"`` (default) — raise :class:`ValueError`.
    - ``"degrade_resolution"`` — double ``resolution`` (powers of two) until
      the window covers the range, keeping ``n_bins`` fixed.
    - ``"collapse_bins"`` — shrink ``n_bins`` to the smallest power of two
      whose window (at the original ``resolution``) covers the range.

    Returns ``(z_lo, n_bins, resolution)`` with the possibly adjusted bin
    count / resolution. Raises ``ValueError`` when the block has no
    populated cells with a finite quantile range, or on ``fit="raise"``
    overflow.
    """
    bounds = [b for b in (_cell_tail_bounds(d, bottom, top) for d in digests) if b is not None]
    if not bounds:
        raise ValueError("chunk has no populated cells with a finite quantile range")

    lo_min = min(b[0] for b in bounds)
    hi_max = max(b[1] for b in bounds)
    z_lo = math.floor(lo_min)
    z_hi = math.ceil(hi_max)
    needed = z_hi - z_lo
    window = n_bins * resolution

    if fit == "collapse_bins":
        # Only ever reduces the bin count, so it cannot help a range that
        # already exceeds the full n_bins window.
        if needed > window:
            raise ValueError(
                f'fit="collapse_bins" cannot grow the window: trimmed span {needed} '
                f"exceeds {n_bins} bins × {resolution} = {window}"
            )
        # Largest power of two ≤ n_bins (the collapsed count is always pow2).
        n = 1 << (int(n_bins).bit_length() - 1)
        while n // 2 >= 1 and (n // 2) * resolution >= needed:
            n //= 2
        return float(z_lo), n, resolution

    if needed <= window:
        return float(z_lo), n_bins, resolution

    if fit == "raise":
        raise ValueError(
            f"trimmed z-range [{z_lo}, {z_hi}] (span {needed}) exceeds the fixed "
            f"window {n_bins} bins × {resolution} = {window}; pass "
            f'fit="degrade_resolution" or fit="collapse_bins" to adapt'
        )
    if fit == "degrade_resolution":
        res = resolution
        while needed > n_bins * res:
            res *= 2.0
        return float(z_lo), n_bins, res
    raise ValueError(f"unknown fit mode {fit!r}")


# --------------------------------------------------------------------------- #
# occupancy (moczarr's own coverage machinery — never zagg's)
# --------------------------------------------------------------------------- #


def _read_store_object(store: Store, key: str) -> bytes | None:
    """Raw bytes of one store object (the ``coverage.moc`` sidecar), or None."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    buf = sync(store.get(key, prototype=default_buffer_prototype()))
    return None if buf is None else buf.to_bytes()


def _coverage_occupancy(store: Store) -> tuple[str, dict, bytes] | None:
    """``(encoding, coverage, sidecar_bytes)`` when the store has EXACT occupancy.

    ``None`` for every store whose mask must degrade to the 2-state
    populated/not channel: no commit stamp (every flat store), a box-only
    envelope, or a bitmap envelope whose sidecar object is gone. Shared by
    :func:`has_exact_occupancy` and the mask build, so the public predicate
    cannot drift from the mask the reader produces.
    """
    try:
        root = zarr.open_group(store, mode="r", zarr_format=3)
    except (FileNotFoundError, KeyError):
        return None
    coverage = parse_leaf_coverage(root.attrs.get(COMMIT_ATTR))
    if coverage is None:
        return None
    encoding = coverage.get("encoding")
    if encoding == "full":
        return "full", coverage, b""  # a full subtree needs no sidecar (D14)
    if encoding != "bitmap" or not coverage.get("sidecar"):
        return None
    payload = _read_store_object(store, str(coverage["sidecar"]))
    if payload is None:
        return None
    return "bitmap", coverage, payload


def _cells_order(words: np.ndarray, field: str, start: int) -> int:
    """HEALPix order of a chunk's written cell words (the cells-axis order)."""
    written = words[words != 0]
    if written.size == 0:
        raise ValueError(
            f"chunk at cell {start} of {field!r} has ragged payloads but no written "
            f"'morton' coordinate — the dense coordinate write did not cover it"
        )
    return decimal_order(morton_decimal(int(written[0])))


def _chunk_word(words: np.ndarray, field: str, start: int) -> int:
    """A read chunk's/block's coverage-cell morton id from its cells' words.

    The written cells' common ancestor at ``cell_order - log4(len(words))``.
    Any span of a nested-ordered cells axis shares that ancestor by
    construction, including a block coarser than the stored shard; this
    raises only when the written cells do NOT share it — the span's
    ``morton`` coordinate is not nested-ordered (not a zagg-written axis).
    """
    written = words[words != 0]
    cell_order = _cells_order(words, field, start)
    order = cell_order - (int(len(words)).bit_length() - 1) // 2  # log4(len)
    ancestors = clip2order(order, written)
    if np.any(ancestors != ancestors[0]):
        raise ValueError(
            f"cells of the block at {start} of {field!r} span more than one "
            f"order-{order} ancestor — the 'morton' coordinate is not in nested "
            f"order over this span, so a cells-axis index does not name a "
            f"position in the block's subtree"
        )
    return int(ancestors[0])


def _load_occupancy(store: Store, arr, words: np.ndarray, field: str) -> tuple | None:
    """The leaf's exact cell occupancy for the mask channel, or ``None``.

    Returns ``("full", None)`` for a fully occupied subtree, ``("bitmap",
    sorted_words)`` with the decoded occupied cell words
    (:func:`moczarr.coverage.decode_bitmap` — the frozen bitmap convention,
    exact or raise), or ``None`` when the store carries no exact occupancy
    (the degrade — see :func:`has_exact_occupancy`). ``words`` are a
    populated chunk's written morton words: the shard id is their ancestor
    at ``cell_order - log4(n_cells)`` — a leaf's cells axis is exactly one
    shard subtree, which is also what binds the bitmap's bit positions to
    the axis.
    """
    found = _coverage_occupancy(store)
    if found is None:
        return None
    encoding, coverage, payload = found
    if encoding == "full":
        return "full", None
    cell_order = _cells_order(words, field, 0)
    if int(coverage.get("cell_order", -1)) != cell_order:
        raise ValueError(
            f"{field!r} coverage envelope measured occupancy at order "
            f"{coverage.get('cell_order')} but the cells axis is order {cell_order} — "
            f"the occupancy bitmap cannot be aligned to the tensor cells"
        )
    n = int(arr.shape[0])
    depth = (n.bit_length() - 1) // 2
    if 4**depth != n:
        raise ValueError(
            f"{field!r} carries a leaf coverage stamp but its {n}-cell axis is not a "
            f"power-of-four shard subtree — the occupancy bitmap cannot be bound to "
            f"a shard"
        )
    written = words[words != 0]
    shard = int(clip2order(cell_order - depth, written[:1])[0])
    return "bitmap", decode_bitmap(payload, shard, cell_order)


def _block_mask(words: np.ndarray, occupancy: tuple | None, block_depth: int) -> np.ndarray:
    """The block's ``(side, side)`` uint8 occupancy base (values 0/1).

    ``1`` marks a cell the leaf's occupancy sidecar records as observed; the
    caller upgrades digest-bearing cells to ``2``. The base is
    stratum-agnostic — which is what makes ``1`` symmetric across the strata
    fields, an observed cell with both strata empty included. Without occupancy truth
    the base stays ``0`` everywhere (the 2-state degrade — ``0`` then
    asserts nothing about observation; :func:`has_exact_occupancy` is the
    discriminator).
    """
    side = 1 << block_depth
    mask = np.zeros((side, side), dtype=np.uint8)
    if occupancy is None:
        return mask
    kind, occupied = occupancy
    if kind == "full":
        mask[:] = 1
        return mask
    if occupied.size:
        idx = np.searchsorted(occupied, words)
        hit = (occupied[np.minimum(idx, occupied.size - 1)] == words) & (words != 0)
        rows, cols = rank_to_rowcol(np.flatnonzero(hit), block_depth)
        mask[rows, cols] = 1
    return mask


def _tensor_side(arr, field: str) -> tuple[int, int]:
    """``(side, depth)`` of one read chunk's square block.

    ``side = 2**depth``: the deinterleave (mortie spec §8) is defined over
    power-of-four subtrees, so a chunk that is not one cannot form a
    spatially faithful ``(side, side)`` block.
    """
    cells_per_chunk = int(arr.chunks[0])
    side = math.isqrt(cells_per_chunk)
    if side * side != cells_per_chunk or side & (side - 1):
        raise ValueError(
            f"{field!r} read chunk holds {cells_per_chunk} cells — not a power-of-"
            f"four subtree, so it cannot deinterleave to a (side, side, n_bins) tensor"
        )
    return side, side.bit_length() - 1


# --------------------------------------------------------------------------- #
# public readers
# --------------------------------------------------------------------------- #


def read_tensors(
    store: Store,
    field: str,
    *,
    n_bins: int = 128,
    resolution: float = 0.5,
    bottom: float = 0.05,
    top: float = 0.95,
    fit: FitMode = "raise",
    dtype: TensorDtype = "uint32",
    block_order: int | None = None,
    max_block_bytes: int = 2 * 1024**3,
    zarr_format: Literal[2, 3] = 3,
) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[float, float], int]]:
    """Yield ``(tensor, mask, (offset, gain), morton_index)`` per coverage block.

    The englacial/zagg#336 reader contract, one tuple per populated block of
    a t-digest field. Sweeps the field's vlen array one read chunk (one
    square cell block) at a time, visiting only the STORED objects
    (:func:`moczarr.ragged.iter_populated_chunks`). Per block: trim each
    cell's tails, derive one shared z-window (:func:`chunk_z_range`),
    rasterize every populated cell (:func:`rasterize_cell`), and place cells
    by the bit deinterleave of their nested rank (:func:`rank_to_rowcol`).

    ``block_order`` assembles the ``4**(chunk_order - block_order)`` read
    chunks of one block-order subtree into a single ``(2**d, 2**d, n_bins)``
    tensor (``d = cell_order - block_order``) with the z-window and ``fit``
    policy reconciled **block-wide** — one shared offset/gain per block.

    The memory bound is per BLOCK: the block's decoded digests plus the
    emitted ``4**block_depth * n_bins`` tensor, which grows 4× per coarser
    order. ``max_block_bytes`` (2 GiB default) refuses the tensor with a
    pointed error naming the size instead of dying in the allocator; it is
    checked against the REQUESTED ``n_bins`` (``fit`` only ever shrinks it).

    Parameters
    ----------
    store : Store
        Zarr store holding the ragged vlen array (a hive leaf store, or a
        flat store root).
    field : str
        Array path (e.g. ``"6/h_tdigest_signal"``).
    n_bins : int, optional
        Number of z-bins (default 128).
    resolution : float, optional
        Bin width in value units (default 0.5).
    bottom, top : float, optional
        Lower/upper density-trim quantiles (default 0.05 / 0.95).
    fit : {"raise", "degrade_resolution", "collapse_bins"}, optional
        Behaviour when the trimmed range exceeds ``n_bins * resolution``
        (default ``"raise"``).
    dtype : {"uint16", "uint32", "float32"}, optional
        Output tensor dtype (default ``"uint32"``). Integer dtypes round
        counts; ``float32`` keeps fractions. A per-bin count exceeding the
        dtype's max wraps on cast — keep ``uint32`` for dense cells.
    block_order : int, optional
        HEALPix order of the emitted blocks (default ``None`` — one block
        per read chunk). Must be at or coarser than the chunk order; a block
        is assembled from whole read chunks with one shared z-window.
    max_block_bytes : int, optional
        Refuse a block whose emitted tensor would exceed this many bytes
        (default 2 GiB). Raise it deliberately to allow a bigger block.
    zarr_format : int, optional
        Zarr format version (default 3).

    Yields
    ------
    (tensor, mask, (offset, gain), morton_index) : (ndarray, ndarray, tuple, int)
        ``tensor`` has shape ``(side, side, n_bins_out)`` and the requested
        dtype. ``mask`` is the block's ``(side, side)`` uint8 occupancy
        channel — ``0`` unobserved, ``1`` observed but no stored digest,
        ``2`` observed with data — from the leaf's ``coverage.moc``
        occupancy sidecar, read WITHOUT touching digest bytes; without one
        the mask degrades to 2-state ``{0, 2}`` (check
        :func:`has_exact_occupancy` before keying on ``mask == 1``).
        ``(offset, gain)`` is the block's shared z-window ``(z_lo,
        resolution)``: bin ``i`` of every cell covers ``[offset + i*gain,
        offset + (i+1)*gain)``. ``morton_index`` is the block's
        coverage-cell morton id.

    Raises
    ------
    ValueError
        On an unknown ``dtype``/``fit``, the strict ragged attrs gate, a
        missing ``morton`` sibling, an out-of-range ``block_order``, a block
        tensor over ``max_block_bytes``, a corrupt/misaligned occupancy
        sidecar, or (with ``fit="raise"``) a window overflow.
    ImportError
        When zagg's digest algebra is not installed (``moczarr[zagg]``).
    """
    if dtype not in _TENSOR_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(_TENSOR_DTYPES)}")
    out_dtype = _TENSOR_DTYPES[dtype]
    is_float = np.issubdtype(out_dtype, np.floating)

    arr, element = open_ragged(store, field, zarr_format=zarr_format)
    morton = _morton_words(store, field, zarr_format)
    side, depth = _tensor_side(arr, field)
    cells_per_chunk = side * side

    chunks = iter_populated_chunks(arr)
    first = next(chunks, None)
    if first is None:
        return
    first_words = np.asarray(morton[first[0] : first[0] + cells_per_chunk])
    occupancy = _load_occupancy(store, arr, first_words, field)
    if block_order is None:
        block_cells, block_depth = cells_per_chunk, depth
    else:
        # The cells-axis order comes from the first populated chunk's words;
        # the block subtree must be whole read chunks (coarser or equal).
        cell_order = _cells_order(first_words, field, first[0])
        chunk_order = cell_order - depth
        if not 0 <= int(block_order) <= chunk_order:
            raise ValueError(
                f"block_order {block_order} is out of range: a block assembles whole "
                f"read chunks, so it must be between 0 and the chunk order "
                f"{chunk_order} (block_order=None reads per chunk)"
            )
        block_depth = cell_order - int(block_order)
        block_cells = 4**block_depth
        if int(arr.shape[0]) % block_cells:
            raise ValueError(
                f"{field!r} has {int(arr.shape[0])} cells — not a whole number of "
                f"order-{block_order} blocks ({block_cells} cells each), so the cells "
                f"axis cannot assemble at this block order"
            )
    block_side = 1 << block_depth
    # Bound the emitted tensor BEFORE allocating it (n_bins is the requested
    # count — the fit policy only ever shrinks it, so this is the bound).
    block_bytes = 4**block_depth * int(n_bins) * out_dtype.itemsize
    if block_bytes > int(max_block_bytes):
        asked = (
            f"block_order={block_order}" if block_order is not None else "the read chunk geometry"
        )
        raise ValueError(
            f"{asked} emits a {block_side}×{block_side}×{n_bins} {dtype} block tensor "
            f"= {block_bytes} bytes ({block_bytes / 1024**3:.2f} GiB), over the "
            f"{int(max_block_bytes)}-byte max_block_bytes limit; use a finer "
            f"block_order, fewer n_bins, or raise max_block_bytes deliberately"
        )

    for block, group in groupby(chain([first], chunks), key=lambda c: c[0] // block_cells):
        bstart = block * block_cells
        cells = [
            (start - bstart + pos, decode_cell(raw, element))
            for start, populated in group
            for pos, raw in populated
        ]
        z_lo, n_bins_c, resolution_c = chunk_z_range(
            [digest for _rank, digest in cells],
            n_bins=n_bins,
            resolution=resolution,
            bottom=bottom,
            top=top,
            fit=fit,
        )

        words = np.asarray(morton[bstart : bstart + block_cells])
        tensor = np.zeros((block_side, block_side, n_bins_c), dtype=out_dtype)
        mask = _block_mask(words, occupancy, block_depth)
        for rank, digest in cells:
            counts = rasterize_cell(digest, z_lo, resolution_c, n_bins_c)
            if not is_float:
                counts = np.rint(counts)
            row, col = rank_to_rowcol(rank, block_depth)
            tensor[row, col, :] = counts.astype(out_dtype)
            mask[row, col] = 2

        yield tensor, mask, (float(z_lo), float(resolution_c)), _chunk_word(words, field, bstart)


def has_exact_occupancy(store: Store) -> bool:
    """Whether this store's :func:`read_tensors` mask is the 3-state channel.

    The mask's two regimes are indistinguishable from the yielded array (a
    degraded mask is ``{0, 2}``, and so is a 3-state mask over a block with
    no observed-but-empty cell), so this is the discriminator — call it once
    per store before keying on the mask's semantics:

    - ``True`` — the leaf carries exact ``coverage.moc`` occupancy, so ``0``
      really means *unobserved* and ``1`` (observed, but no digest stored on
      the field being read — whichever stratum that is) is reported wherever
      it occurs.
    - ``False`` — no exact occupancy (no commit stamp, i.e. every flat
      store; a box-only envelope; or a missing sidecar): the mask degrades
      to 2-state populated/not and ``0`` means only "no stored digest here".

    One attrs read plus (for a bitmap envelope) the small sidecar object; no
    digest bytes. Shares :func:`_coverage_occupancy` with the mask build, so
    it cannot report a regime the reader does not produce.
    """
    return _coverage_occupancy(store) is not None

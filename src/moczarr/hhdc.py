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
  held by the committed goldens plus ``TestLiveParity``. That live leg needs
  zagg's post-englacial/zagg#339 reader surface: **zagg 0.40.0 is the first
  release to carry it** (0.39.0's reader predates the deinterleave, so the
  leg silently skipped against the declared ``zagg>=0.39`` floor — the
  goldens were the only enforcement). Verified against the 0.40.0 sdist; the
  extra's floor is still ``>=0.39``, and bumping it to ``>=0.40`` — which
  turns the parity legs from skip-guarded into always-on — is a dependency
  change awaiting sign-off, not something this module can assume.
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
from mortie import clip2order, orders_of, rank_to_xy, xy_to_rank
from zarr.abc.store import Store

from moczarr.convention import (
    COMMIT_ATTR,
    is_point_word,
    morton_decimal,
    morton_word,
    point_to_area29,
)
from moczarr.coverage import decode_bitmap, parse_leaf_coverage
from moczarr.ragged import (
    _cells_order,
    _morton_words,
    _refuse_sub_chunk,
    _subtree_span,
    decode_cell,
    iter_populated_chunks,
    open_ragged,
    stored_chunk_spans,
)

__all__ = [
    "block_rank",
    "cell_index",
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
# word addressing (mortie spec §1 packed-word geometry)
# --------------------------------------------------------------------------- #

#: Spec §1 packed-word geometry: ``[4-bit prefix | 54-bit body | 6-bit suffix]``.
#: The body carries one 2-bit digit per level for levels 1..27, most
#: significant first; levels 28 and 29 have no body room and ride the
#: suffix's parent-first preorder band instead (see :func:`block_rank`).
_SUFFIX_BITS = 6
_BODY_LEVELS = 27
_PREORDER_MIN = 28
_MAX_ORDER = 29
_DIGIT_MASK = np.uint64(0x3)
_SUFFIX_MASK = np.uint64(0x3F)


def block_rank(words, block_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-local nested rank of each word, and each word's own order.

    The decode a **located companion** needs (spec §9): the companion
    carries one morton word per observation and has no cells axis to index,
    so a reader placing those observations inside an order-``block_order``
    block has to recover each word's nested rank *within that block* itself.
    On the tensor path a cell's rank IS its position on the cells axis and
    no word is ever decoded — this is the same quantity for the path where
    it is not handed to you. Pairs with :func:`rank_to_rowcol`, one
    vectorized call per depth (a real companion is mixed-order, so a single
    call on ``order[0]`` would raise *rank must lie in [0, 4**d)* on every
    word of a different order)::

        rank, order = block_rank(located_words, block_order)
        for depth in np.unique(order - block_order):
            sel = order - block_order == depth
            row, col = rank_to_rowcol(rank[sel], int(depth))

    Point words normalize first (:func:`moczarr.convention.point_to_area29`
    — the §1 suffix re-base ``48 + t28*4 + t29`` ->
    ``28 + t28*5 + (t29 + 1)``), so a located companion's order-29 POINT
    words decode through the same area arithmetic as every other word;
    skipping that step reads the level-28/29 digits out of their bands.
    Mixed orders in one array are supported — a leaf mixes order-29 located
    words with coarser cell words — which is why the per-word order is
    returned: group by ``order - block_order`` and make one vectorized
    :func:`rank_to_rowcol` call per depth.

    Vectorized over the whole array: the decode loops over the ≤29 LEVELS,
    never over the words (a located companion is millions of words per
    leaf, so a per-word :func:`moczarr.morton_decimal` loop is not an
    option). The digit extraction is the spec §1 geometry — levels 1..27
    read 2 bits out of the body, level 28 is ``(suffix - 28) // 5`` and
    level 29 is ``(suffix - 28) % 5 - 1`` — and the block-local rank is
    ``sum(digit_L * 4**(order - L))`` over the levels below the block.

    Parameters
    ----------
    words : array-like
        Packed ``uint64`` morton words (AREA or POINT, any mix of orders).
        The ``0`` FILL word is REFUSED at every ``block_order``, not passed
        through: it is not a word (the §1 prefix nibble is base cell + 1, so
        prefix ``0`` is unreachable), and at ``block_order == 0`` it would
        otherwise rank as a legitimate ``(0, 0)`` — a sentinel
        indistinguishable from data. A caller handing over a fill-padded
        ``morton`` coordinate masks it first, as :func:`_cells_order` and
        :func:`_chunk_word` do.
    block_order : int
        HEALPix order of the enclosing block — the subtree the returned rank
        is local to. ``0`` ranks a word within its whole base cell.

    Returns
    -------
    (rank, order) : (ndarray, ndarray)
        ``rank`` is ``uint64``, in ``[0, 4**(order - block_order))``;
        ``order`` is ``int64``, the word's OWN encoded order (29 for point
        words — their encoded order, not a clip). Both keep the input's
        shape, so a scalar word in yields 0-d arrays out.

    Raises
    ------
    ValueError
        When ``block_order`` is negative; when any word is the ``0`` FILL
        word (see above); or when ``block_order`` is finer than any word's
        own order — that word lies at or above the block and has no position
        inside it, so the rank would have to be truncated rather than
        computed.
    """
    packed = np.asarray(point_to_area29(np.asarray(words, dtype=np.uint64)), dtype=np.uint64)
    block = int(block_order)
    if block < 0:
        raise ValueError(f"block_order {block_order} is negative (a block order is 0..29)")
    # Before the order check, so the fill word is diagnosed as the fill word
    # rather than as a shallow word the block is finer than.
    fill = int(np.count_nonzero(packed == np.uint64(0)))
    if fill:
        raise ValueError(
            f"{fill} of the {packed.size} word(s) given is the 0 FILL word, not a morton "
            f"word: the §1 prefix nibble encodes base cell + 1, so 0 names no cell "
            f"(mortie's own morton_decimal refuses it). A companion's 'morton' coordinate "
            f"is fill-padded over its unwritten rows — mask them out (words[words != 0]) "
            f"before ranking"
        )
    order = np.asarray(orders_of(packed), dtype=np.int64).reshape(packed.shape)
    if np.any(order < block):
        shallow = order[order < block]
        raise ValueError(
            f"block_order {block} is finer than {shallow.size} of the {order.size} "
            f"word(s) given (shallowest order {int(shallow.min())}): such a word lies "
            f"at or above the block, so it has no position INSIDE it and its rank "
            f"cannot be computed (only truncated)"
        )
    suffix = (packed & _SUFFIX_MASK).astype(np.int64)
    rank = np.zeros(packed.shape, dtype=np.uint64)
    for level in range(block + 1, _MAX_ORDER + 1):
        active = order >= level
        if not active.any():
            break  # orders are fixed and levels ascend: nothing deeper is active
        if level <= _BODY_LEVELS:
            shift = np.uint64(_SUFFIX_BITS + 2 * (_BODY_LEVELS - level))
            digit = ((packed >> shift) & _DIGIT_MASK).astype(np.int64)
        elif level == _BODY_LEVELS + 1:
            digit = (suffix - _PREORDER_MIN) // 5
        else:
            digit = (suffix - _PREORDER_MIN) % 5 - 1
        # Inactive words contribute nothing; their place/digit would be
        # meaningless (a negative shift, or the preorder band's -1 filler).
        place = np.where(active, 2 * (order - level), 0).astype(np.uint64)
        rank += np.where(active, digit, 0).astype(np.uint64) << place
    return rank, order


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
    subtree: int | str | None = None,
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
        is assembled from whole read chunks with one shared z-window. With a
        ``subtree`` the floor is the order of the span actually VISITED —
        the subtree CLIPPED to this axis — so the composed bound is
        ``max(subtree_order, axis_root_order) <= block_order <=
        chunk_order``: blocks tile the visited span, and a word above the
        axis root (which clips to the whole axis) takes the ROOT's order as
        its floor, not its own.
    subtree : int or str, optional
        Restrict the sweep to the read chunks below this morton ancestor —
        packed area word or decimal string, as in
        :func:`moczarr.ragged.read_ragged` (issue #29; zagg spec §1.5):
        only stored objects overlapping the subtree's cell span are
        fetched. The subtree must be at or coarser than the READ-CHUNK
        order — a finer word raises pointing at
        :func:`moczarr.ragged.read_cell` (the ratified v1 refusal: a
        sub-chunk block would re-derive its z-window from fewer cells and
        stop being a slice of the chunk tensor; recover a sub-chunk region
        client-side as a corner slice of the chunk tensor, keeping its
        shared ``(offset, gain)``). A well-formed word disjoint from this
        axis warns at most once per call (naming the word and the axis
        root) and yields nothing — so an **empty yield is ambiguous**
        between "in-domain, nothing stored" and "outside the domain"; the
        warning is the only discriminator, and its DELIVERY is the caller's
        ``warnings`` filter's call (the default action dedups a repeat of
        the same warning in the same process). Malformed / too-deep words
        raise ``ValueError``.
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
        sidecar, a ``subtree`` finer than the read chunks (or malformed /
        too-deep), or (with ``fit="raise"``) a window overflow.
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

    span, spans = _subtree_span(arr, morton, field, subtree)
    _refuse_sub_chunk(span, subtree, field, cells_per_chunk)
    chunks = iter_populated_chunks(arr, span=span, spans=spans)
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
        min_order = 0
        if span is not None:
            # Blocks must tile INSIDE the subtree span (subtree_order ≤
            # block_order): a coarser block would reach past the span and
            # label a partial assembly with the bigger block's word.
            min_order = cell_order - ((span[1] - span[0]).bit_length() - 1) // 2
        if not min_order <= int(block_order) <= chunk_order:
            raise ValueError(
                f"block_order {block_order} is out of range: a block assembles whole "
                f"read chunks within the visited span, so it must be between "
                f"{min_order} and the chunk order {chunk_order} "
                f"(block_order=None reads per chunk)"
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


def _read_chunk_id(morton_index: int | str) -> tuple[int, str]:
    """``(word, decimal)`` of a read-chunk id, or a ValueError naming the caller.

    The validation :func:`moczarr.convention.normalize_subtree` performs for
    ``subtree=``, applied to :func:`cell_index`'s ``morton_index``: a bare
    :func:`moczarr.convention.morton_word` passes any int straight through,
    so a malformed one reached the span scan and came back reported as a
    missing chunk — the store blamed for a caller's mistake. Validity is
    checked before kind, so an id whose §1 prefix nibble is ``0`` (a decimal
    id typed as an int, most of the time) is called what it is rather than
    misread as a POINT word off its suffix band.
    """
    word = morton_word(morton_index) if isinstance(morton_index, str) else int(morton_index)
    if not 0 <= word < 2**64:
        raise ValueError(
            f"morton_index {morton_index!r} is outside the uint64 range, not a packed "
            f"morton word; parse a decimal id by passing it as a string instead"
        )
    try:
        decimal = morton_decimal(word)
    except ValueError as exc:
        raise ValueError(
            f"morton_index {morton_index!r} is not a valid packed morton word; parse "
            f"a decimal id by passing it as a string instead (an int argument is read "
            f"as a packed word)"
        ) from exc
    if is_point_word(word):
        raise ValueError(
            f"morton_index {morton_index!r} is an order-29 POINT word: a read-chunk id "
            f"names an AREA subtree (spec §1/§4 — points have no descendants)"
        )
    return word, decimal


def cell_index(
    store: Store,
    field: str,
    morton_index: int | str,
    row: int,
    col: int,
    *,
    zarr_format: Literal[2, 3] = 3,
) -> int:
    """Global cells-axis index of a reported ``(row, col)`` — the :func:`read_cell` key.

    The sweep readers report a CHUNK-LOCAL position, while
    :func:`moczarr.read_cell` addresses the array's GLOBAL cells axis:
    :func:`rowcol_to_rank` inverts the deinterleave back to a rank ``0..4**depth
    - 1`` *within the chunk*, and the chunk's own start offset is the missing
    term — so feeding a bare rank to :func:`moczarr.read_cell` silently reads
    the wrong cell (it is always in range, so nothing complains). This resolves
    the offset from the sibling ``morton`` coordinate and returns ``chunk_start
    + rowcol_to_rank(row, col, depth)``.

    ``morton_index`` is a READ-CHUNK id — the id :func:`moczarr.read_ragged`
    and the default (per-chunk, ``block_order=None``) :func:`read_tensors`
    report — as a packed area word or a decimal string. Both currencies are
    validated BEFORE the store is searched, the way
    :func:`moczarr.convention.normalize_subtree` validates ``subtree=``: an
    int outside the uint64 range, an int that is not a valid packed word,
    and an order-29 POINT word each get their own error naming the CALLER's
    mistake, rather than reaching the span scan and coming back blaming the
    store. The one case no guard can catch — a decimal id typed as an int
    that happens to have a legal prefix nibble — is why the not-found
    message renders BOTH currencies of what it parsed. A coarser
    ``block_order`` block id names no single chunk and raises. Only the
    array's STORED spans are searched
    (:func:`moczarr.ragged.stored_chunk_spans`, the same objects the sweep
    readers visit), one small slice of the ``morton`` coordinate per span —
    never the whole axis, and no digest bytes.

    Raises
    ------
    ValueError
        If ``morton_index`` is not a well-formed AREA read-chunk id in
        either currency, if ``row``/``col`` are outside the chunk's
        ``(side, side)`` block, or if no stored chunk carries
        ``morton_index``.
    """
    arr, _element = open_ragged(store, field, zarr_format=zarr_format)
    morton = _morton_words(store, field, zarr_format)
    side, depth = _tensor_side(arr, field)
    cells_per_chunk = side * side
    if not (0 <= int(row) < side and 0 <= int(col) < side):
        raise ValueError(f"({row}, {col}) is outside {field!r}'s ({side}, {side}) read-chunk block")
    rank = int(rowcol_to_rank(int(row), int(col), depth))
    target, decimal = _read_chunk_id(morton_index)
    for span_start, span_stop in stored_chunk_spans(arr):
        span_words = morton[span_start:span_stop]
        for offset in range(0, span_stop - span_start, cells_per_chunk):
            words = span_words[offset : offset + cells_per_chunk]
            start = span_start + offset
            if not np.any(words) or _chunk_word(words, field, start) != target:
                continue
            return start + rank
    raise ValueError(
        f"no stored read chunk of {field!r} carries morton id {decimal} (word "
        f"{target}) — cell_index resolves the READ-CHUNK ids the sweep readers "
        f"report (a coarser block_order block id names no single chunk; and an "
        f"int argument is read as a PACKED word, so a decimal id passed as an "
        f"int resolves to the unrelated id shown here)"
    )

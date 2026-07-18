"""Interval substrate for the MOC-backed lazy index — numpy-only, no I/O.

The lazy index (phase 5, espg/moczarr#1) holds ``open_hive``'s row domain as
a set of intervals instead of a materialized ``morton`` coordinate. zagg
leaves are dense within a shard, so the domain is pure arithmetic: shard word
→ one contiguous subtree run at ``cell_order``, AOI intersection → interval
intersection. No cell arrays are read to know the coordinate.

**Rank space, not word space** (the phase-5 probe finding): packed morton
words at a fixed cell order are NOT unit-stride across a shard subtree —
mortie packs the base+digit field MSB-aligned with an order marker in the
low bits, so consecutive rank tails differ by a large power-of-two stride.
But ``word >> shift`` (the base+digit field alone) IS unit-stride: it is a
global rank coordinate ``K`` at the cell order, contiguous across
rank-consecutive shard subtrees within a base cell (verified across sibling
and parent-crossing boundaries) and monotonic everywhere words are. All
intervals here therefore live in K space, inclusive ``[lo, hi]``; words
convert at the boundaries via ``word = (K << shift) | marker``. The
``(shift, marker)`` pair is derived empirically from mortie's own packing at
construction (two ``morton_word`` calls, cached) — never hardcoded — and
subtree spans are re-checked against ``4**depth``, so packing drift raises
instead of mis-ranking.

Single cell order by construction (a store's cell coordinate is one order);
mixed-order domains (pyramids, zagg#262) are the named phase-5 seam —
intervals-per-order is the natural extension, gated with issue #8 on
mortie#116.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from moczarr.convention import (
    decimal_base,
    decimal_order,
    decimal_rank,
    morton_decimal,
    morton_word,
)


@lru_cache(maxsize=None)
def _shift_and_marker(cell_order: int) -> tuple[int, int]:
    """``(shift, marker)`` of packed words at ``cell_order``.

    ``shift`` positions the base+digit field (``K = word >> shift``);
    ``marker`` is the constant low-bit residue every word at this order
    carries (``word == (K << shift) | marker``). Derived from mortie's own
    packing — the stride between the two lowest rank tails — and validated
    as a power of two, so a packing change fails loudly here rather than
    silently mis-ranking.
    """
    if not 0 <= cell_order <= 29:
        raise ValueError(f"cell_order {cell_order} outside the packed range 0..29")
    if cell_order == 0:
        w0, w1 = morton_word("1"), morton_word("2")
    else:
        w0 = morton_word("1" + "1" * cell_order)
        w1 = morton_word("1" + "1" * (cell_order - 1) + "2")
    stride = w1 - w0
    if stride <= 0 or stride & (stride - 1):
        raise AssertionError(
            f"mortie packing drift: rank stride {stride} at order {cell_order} "
            f"is not a positive power of two — the rank-space substrate no longer holds"
        )
    shift = stride.bit_length() - 1
    return shift, w0 & (stride - 1)


def _coalesce(intervals: np.ndarray) -> np.ndarray:
    """Sorted, disjoint, adjacency-merged ``(n, 2)`` inclusive K intervals.

    Overlapping and rank-adjacent inputs merge (set-union semantics — AOI
    covers may overlap); a reversed pair raises. Comparisons stay in uint64
    (guarded subtractions, no signed casts) so order-29 K values near 2**64
    stay exact.
    """
    intervals = np.asarray(intervals, dtype=np.uint64).reshape(-1, 2)
    if intervals.size == 0:
        return intervals
    if (intervals[:, 1] < intervals[:, 0]).any():
        raise ValueError("reversed interval (hi < lo)")
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    hi_cummax = np.maximum.accumulate(intervals[:, 1])
    starts = np.empty(len(intervals), dtype=bool)
    starts[0] = True
    # New group iff lo > running_hi + 1, computed as two uint64-safe compares.
    gap = intervals[1:, 0] > hi_cummax[:-1]
    gap &= (intervals[1:, 0] - hi_cummax[:-1]) > 1
    starts[1:] = gap
    first = np.flatnonzero(starts)
    last = np.append(first[1:], len(intervals)) - 1
    return np.stack([intervals[first, 0], hi_cummax[last]], axis=1)


class MortonRanges:
    """A sorted disjoint interval set over morton cells at one fixed order.

    The four ops the lazy index needs — membership, rank, positional take,
    interval intersect — are all searchsorted/cumsum arithmetic on the
    ``(n, 2)`` K-space ``intervals`` array. The concatenated ascending run of
    all covered cells is the *domain*: ``rank`` maps words to domain
    positions, ``take`` maps positions back, ``fabricate`` materializes the
    whole thing (the only op that allocates O(size)).
    """

    __slots__ = ("intervals", "cell_order", "_shift", "_marker", "_offsets")

    def __init__(self, intervals, cell_order: int):
        self.cell_order = int(cell_order)
        self._shift, self._marker = _shift_and_marker(self.cell_order)
        self.intervals = _coalesce(intervals)
        sizes = self.intervals[:, 1] - self.intervals[:, 0] + np.uint64(1)
        self._offsets = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(sizes)])

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_shards(cls, shard_words, cell_order: int) -> "MortonRanges":
        """Subtree runs of shard words: one interval per shard at ``cell_order``.

        Shards may sit at any order <= ``cell_order`` (each word carries its
        own); duplicates and rank-adjacent shards merge. Every subtree span
        is re-checked against ``4**depth`` (the packing-drift guard).
        """
        shift, _ = _shift_and_marker(cell_order)
        intervals = []
        for word in np.unique(np.asarray(shard_words, dtype=np.uint64).ravel()):
            decimal = morton_decimal(int(word))
            depth = int(cell_order) - decimal_order(decimal)
            if depth < 0:
                raise ValueError(
                    f"shard {decimal} sits below cell_order {cell_order} "
                    f"(order {decimal_order(decimal)})"
                )
            lo = morton_word(decimal + "1" * depth) >> shift
            hi = morton_word(decimal + "4" * depth) >> shift
            if hi - lo + 1 != 4**depth:
                raise AssertionError(
                    f"mortie packing drift: subtree of {decimal} spans {hi - lo + 1} "
                    f"ranks at order {cell_order}, expected {4**depth}"
                )
            intervals.append((lo, hi))
        return cls(np.asarray(intervals, dtype=np.uint64).reshape(-1, 2), cell_order)

    @classmethod
    def from_root_coverage(cls, envelope: dict, cell_order: int) -> "MortonRanges":
        """The domain a root ``"ranges"`` envelope declares, at ``cell_order``.

        One interval per envelope range, WITHOUT the O(covered shards)
        expansion of :func:`moczarr.coverage.ranges_words`: consecutive-rank
        shard subtrees are contiguous in K space, so ``[lo_shard, hi_shard]``
        maps to ``[first cell of lo, last cell of hi]`` directly. Same
        validation posture as ``ranges_words`` — malformed ranges raise (a
        corrupt cache must never yield a plausible partial domain) — plus
        the span re-check as the packing-drift guard.
        """
        order = int(envelope["order"])
        depth = int(cell_order) - order
        if depth < 0:
            raise ValueError(f"envelope order {order} is below cell_order {cell_order}")
        shift, _ = _shift_and_marker(cell_order)
        intervals = []
        for lo_dec, hi_dec in envelope["ranges"]:
            base = decimal_base(lo_dec)
            lo_rank, hi_rank = decimal_rank(lo_dec), decimal_rank(hi_dec)
            ok = decimal_base(hi_dec) == base and lo_rank <= hi_rank
            ok = ok and decimal_order(lo_dec) == order and decimal_order(hi_dec) == order
            if not ok:
                raise ValueError(f"malformed coverage range [{lo_dec}, {hi_dec}] at order {order}")
            lo = morton_word(lo_dec + "1" * depth) >> shift
            hi = morton_word(hi_dec + "4" * depth) >> shift
            if hi - lo + 1 != (hi_rank - lo_rank + 1) * 4**depth:
                raise AssertionError(
                    f"mortie packing drift: range [{lo_dec}, {hi_dec}] spans {hi - lo + 1} "
                    f"ranks at order {cell_order}, expected {(hi_rank - lo_rank + 1) * 4**depth}"
                )
            intervals.append((lo, hi))
        return cls(np.asarray(intervals, dtype=np.uint64).reshape(-1, 2), cell_order)

    @classmethod
    def from_cell_words(cls, words, cell_order: int | None = None) -> "MortonRanges":
        """Run-detect an explicit strictly-ascending cell-word array.

        The materialized→lazy direction (``MortonMocIndex.from_variables``).
        ``cell_order`` defaults to the order the words themselves carry;
        mixed orders or a non-ascending array raise (mirroring ``aoi_mask``'s
        raise-on-ambiguity discipline).
        """
        words = np.asarray(words, dtype=np.uint64).ravel()
        if cell_order is None:
            if words.size == 0:
                raise ValueError("cannot infer cell_order from zero words; pass it explicitly")
            from mortie import infer_order_from_morton

            cell_order = int(infer_order_from_morton(words))
        shift, marker = _shift_and_marker(int(cell_order))
        if words.size == 0:
            return cls(np.empty((0, 2), dtype=np.uint64), cell_order)
        stride = np.uint64(1) << np.uint64(shift)
        if (words & (stride - np.uint64(1)) != np.uint64(marker)).any():
            raise ValueError(f"words are not all at cell_order {cell_order} (mixed orders?)")
        if (words[1:] <= words[:-1]).any():
            raise ValueError("cell words must be strictly ascending")
        ranks = words >> np.uint64(shift)
        breaks = np.flatnonzero(np.diff(ranks) > np.uint64(1))
        first = np.concatenate([[0], breaks + 1])
        last = np.concatenate([breaks, [ranks.size - 1]])
        return cls(np.stack([ranks[first], ranks[last]], axis=1), cell_order)

    # -- interval algebra -------------------------------------------------

    def _clip(self, clip_intervals: np.ndarray) -> "MortonRanges":
        """Intersection with a coalesced K-interval array (the sweep kernel)."""
        los, his = self.intervals[:, 0], self.intervals[:, 1]
        pieces = []
        for clip_lo, clip_hi in clip_intervals:
            k0 = np.searchsorted(his, clip_lo, side="left")  # first with hi >= clip_lo
            k1 = np.searchsorted(los, clip_hi, side="right")  # first with lo > clip_hi
            if k1 > k0:
                pieces.append(
                    np.stack(
                        [np.maximum(los[k0:k1], clip_lo), np.minimum(his[k0:k1], clip_hi)],
                        axis=1,
                    )
                )
        stacked = np.concatenate(pieces) if pieces else np.empty((0, 2), dtype=np.uint64)
        return type(self)(stacked, self.cell_order)

    def _aoi_intervals(self, aoi) -> np.ndarray:
        """An AOI morton cover as coalesced K intervals at ``cell_order``.

        The two-way containment semantics of :func:`moczarr.coverage.aoi_mask`
        in interval space: a coarser-or-equal member covers its whole subtree
        run; a finer member collapses to its containing cell (one rank).
        Members may be mixed-order; strings are accepted.
        """
        values = np.asarray(aoi).ravel() if np.asarray(aoi).ndim else np.asarray([aoi])
        intervals = []
        for value in values:
            word = morton_word(value.item() if hasattr(value, "item") else value)
            decimal = morton_decimal(word)
            depth = self.cell_order - decimal_order(decimal)
            if depth >= 0:
                lo = morton_word(decimal + "1" * depth) >> self._shift
                hi = morton_word(decimal + "4" * depth) >> self._shift
            else:
                from mortie import clip2order

                cell = clip2order(self.cell_order, np.asarray([word], dtype=np.uint64))[0]
                lo = hi = int(cell) >> self._shift
            intervals.append((lo, hi))
        return _coalesce(np.asarray(intervals, dtype=np.uint64).reshape(-1, 2))

    def intersect(self, aoi) -> "MortonRanges":
        """The sub-domain an AOI morton cover keeps — ``aoi_mask`` in interval space."""
        return self._clip(self._aoi_intervals(aoi))

    def intersection(self, other: "MortonRanges") -> "MortonRanges":
        """Set intersection with another interval set at the same order."""
        self._check_same_order(other)
        return self._clip(other.intervals)

    def union(self, other: "MortonRanges") -> "MortonRanges":
        """Set union with another interval set at the same order."""
        self._check_same_order(other)
        return type(self)(np.concatenate([self.intervals, other.intervals]), self.cell_order)

    def _check_same_order(self, other: "MortonRanges") -> None:
        if not isinstance(other, MortonRanges):
            raise TypeError(f"expected MortonRanges, got {type(other).__name__}")
        if other.cell_order != self.cell_order:
            raise ValueError(
                f"mixed cell orders ({self.cell_order} vs {other.cell_order}); "
                f"intervals-per-order is the deferred pyramid seam (issue #8)"
            )

    # -- domain arithmetic ------------------------------------------------

    @property
    def size(self) -> int:
        """Number of cells in the domain."""
        return int(self._offsets[-1])

    def _locate(self, words) -> tuple[np.ndarray, np.ndarray]:
        """``(interval index, member mask)`` of words; index only valid where member."""
        words = np.asarray(words, dtype=np.uint64).ravel()
        if self.intervals.size == 0:
            return np.zeros(words.size, dtype=np.intp), np.zeros(words.size, dtype=bool)
        stride = np.uint64(1) << np.uint64(self._shift)
        ranks = words >> np.uint64(self._shift)
        idx = np.searchsorted(self.intervals[:, 0], ranks, side="right")
        good = idx > 0
        idx = np.maximum(idx, 1) - 1
        good &= ranks <= self.intervals[idx, 1]
        good &= (words & (stride - np.uint64(1))) == np.uint64(self._marker)
        return idx, good

    def member(self, words) -> np.ndarray:
        """Boolean mask: which words the domain contains (order marker included)."""
        _, good = self._locate(words)
        return good

    def rank(self, words, missing: int | None = None) -> np.ndarray:
        """Positions of words in the ascending domain (``int64``).

        Non-members raise ``KeyError`` by default; ``missing=<fill>`` maps
        them to that value instead (the reindex posture).
        """
        words = np.asarray(words, dtype=np.uint64).ravel()
        idx, good = self._locate(words)
        if not good.all() and missing is None:
            bad = words[~good][0]
            raise KeyError(f"{morton_decimal(int(bad))} (word {int(bad)}) is not in the domain")
        if self.intervals.size == 0:
            return np.full(words.size, missing if missing is not None else 0, dtype=np.int64)
        ranks = words >> np.uint64(self._shift)
        positions = (self._offsets[idx] + (ranks - self.intervals[idx, 0])).astype(np.int64)
        if missing is not None:
            positions[~good] = missing
        return positions

    def take(self, positions) -> np.ndarray:
        """Words at domain positions (``uint64``); negatives wrap, OOB raises."""
        positions = np.asarray(positions, dtype=np.int64).ravel()
        positions = np.where(positions < 0, positions + self.size, positions)
        if positions.size and ((positions < 0).any() or (positions >= self.size).any()):
            raise IndexError(f"position out of bounds for domain of size {self.size}")
        upos = positions.astype(np.uint64)
        idx = np.searchsorted(self._offsets, upos, side="right") - 1
        ranks = self.intervals[idx, 0] + (upos - self._offsets[idx])
        return (ranks << np.uint64(self._shift)) | np.uint64(self._marker)

    def subset(self, indexer) -> "MortonRanges":
        """The sub-domain at positional ``indexer`` (slice or integer array).

        A unit-step slice clips intervals in position space (no
        materialization). Integer arrays must be strictly increasing after
        negative-wrap — an interval set has no reordering freedom — so a
        non-monotonic or duplicated indexer raises ``ValueError`` (the index
        layer degrades to dropping the lazy index in that case).
        """
        if isinstance(indexer, slice):
            start, stop, step = indexer.indices(self.size)
            if step == 1:
                if stop <= start:
                    return type(self)(np.empty((0, 2), dtype=np.uint64), self.cell_order)
                lo = self.take([start])[0] >> np.uint64(self._shift)
                hi = self.take([stop - 1])[0] >> np.uint64(self._shift)
                return self._clip(np.asarray([[lo, hi]], dtype=np.uint64))
            indexer = np.arange(start, stop, step, dtype=np.int64)
        positions = np.asarray(indexer, dtype=np.int64).ravel()
        positions = np.where(positions < 0, positions + self.size, positions)
        if positions.size and (np.diff(positions) <= 0).any():
            raise ValueError(
                "an interval set cannot represent a non-increasing or duplicated "
                "positional indexer; materialize the coordinate instead"
            )
        return type(self).from_cell_words(self.take(positions), self.cell_order)

    def fabricate(self) -> np.ndarray:
        """Materialize the whole domain as ascending packed words (``uint64``).

        The one O(size) allocation; byte-equal to the coordinate the eager
        opener builds (the phase-5 money property).
        """
        parts = [np.arange(lo, hi + np.uint64(1), dtype=np.uint64) for lo, hi in self.intervals]
        ranks = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
        return (ranks << np.uint64(self._shift)) | np.uint64(self._marker)

    # -- comparison / repr ------------------------------------------------

    def equals(self, other: object) -> bool:
        """Same domain: same cell order and identical intervals."""
        return (
            isinstance(other, MortonRanges)
            and other.cell_order == self.cell_order
            and np.array_equal(other.intervals, self.intervals)
        )

    def __repr__(self) -> str:
        return (
            f"MortonRanges(order={self.cell_order}, ranges={len(self.intervals)}, size={self.size})"
        )

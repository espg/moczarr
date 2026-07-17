"""Coverage-envelope decoding (``morton-moc/1``), read side — pure functions.

The tiered coverage convention (zagg ``sparse_coverage.md`` §4): a leaf's
commit stamp carries a ``coverage`` envelope — the tier-0 morton box (<= 4
decimal-string members, null-padded) plus an ``encoding`` discriminator:

- ``"full"``   — coverage is the whole shard subtree; no sidecar exists.
- ``"bitmap"`` — exact cell-order occupancy lives in the in-leaf
  ``coverage.moc`` sidecar: a zstd-compressed bit field, bit ``i`` = the
  i-th subtree cell in ascending packed-word order (base-4 digit tail,
  digits ``1..4`` -> ``0..3``), MSB-first per byte.
- absent       — box-only (phase-1 stamps, depth-0 configs).

The store root's ``coverage.moc`` is a ``"ranges"`` envelope: inclusive
``[first, last]`` runs of same-order shard ids within one base cell,
consecutive in digit-tail rank, endpoints as decimal STRINGS (packed words
exceed 2^53 — raw JSON numbers would be float-mangled).

Postures, inherited from the design's D9 discipline: envelopes above the
leaf are caches — an unusable one reads as absent (``None``) and the caller
degrades to the walk, never to a wrong answer. A PRESENT-but-corrupt bitmap
sidecar raises instead: silently zero-padding would fabricate false
negatives, indistinguishable from healthy sparse coverage.
"""

from __future__ import annotations

import numpy as np

from moczarr.convention import (
    decimal_base,
    decimal_order,
    decimal_rank,
    morton_word,
    rank_tail,
)

#: Convention version of coverage envelopes (leaf tier-0/bitmap and root ranges).
COVERAGE_SPEC = "morton-moc/1"
#: Fixed slot count of the tier-0 morton box (1-4 members, null-padded).
COVERAGE_BOX_SLOTS = 4


def parse_leaf_coverage(stamp: object) -> dict | None:
    """The ``coverage`` envelope from a commit stamp, or ``None`` when absent.

    Tolerant by design: debris (``None`` stamp), pre-coverage stamps, a
    malformed payload, or an unknown/future spec all read as absent — the
    box tiers are indexes, never truth, so a reader without them degrades to
    opening the leaf. Strict on the spec gate: a future envelope version
    must be adopted deliberately, not half-parsed.
    """
    if not isinstance(stamp, dict):
        return None
    coverage = stamp.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("spec") != COVERAGE_SPEC:
        return None
    return dict(coverage)


def box_words(coverage: dict) -> np.ndarray:
    """The tier-0 box members as packed ``uint64`` words (nulls dropped).

    Feed to ``mortie.moc_and`` against an AOI cover for the cheap leaf
    reject: the box is a conservative superset (false positives possible,
    false negatives impossible).
    """
    members = [morton_word(s) for s in coverage["box"] if s is not None]
    return np.asarray(members, dtype=np.uint64)


def decode_bitmap(payload: bytes, shard: str | int, cell_order: int) -> np.ndarray:
    """Occupied cell words from a bitmap-sidecar payload — exact, or raise.

    Returns the sorted packed ``uint64`` words at ``cell_order`` whose bits
    are set. A corrupt payload — zstd garbage, or a decompressed size other
    than the deterministic ``ceil(4^depth / 8)`` bytes — raises rather than
    zero-padding to a plausible partial cell set (a false negative; the
    exact truth is intact in the leaf, so surfacing beats under-reporting).
    """
    from numcodecs import Zstd

    from moczarr.convention import morton_decimal

    dec = morton_decimal(shard)
    depth = int(cell_order) - decimal_order(dec)
    if depth <= 0:
        raise ValueError(f"cell_order {cell_order} is not below shard {dec}'s order")
    raw = np.frombuffer(bytes(Zstd().decode(payload)), dtype=np.uint8)
    expected = -(-(4**depth) // 8)
    if raw.size != expected:
        raise ValueError(
            f"coverage sidecar decompressed to {raw.size} B; an order-{cell_order} bitmap "
            f"for shard {dec} is exactly {expected} B — refusing to zero-pad or truncate "
            f"(a partial cell set would be a false negative)"
        )
    bits = np.unpackbits(raw, count=4**depth)
    words = np.empty(int(bits.sum()), dtype=np.uint64)
    for i, rank in enumerate(np.flatnonzero(bits)):
        words[i] = morton_word(dec + rank_tail(int(rank), depth))
    return np.sort(words)


def parse_root_coverage(payload: object) -> dict | None:
    """A usable store-root coverage envelope, or ``None``.

    The root MOC is a regenerable cache: a non-mapping payload, an unknown
    spec, or a non-``"ranges"`` encoding reads as absent and the caller
    falls back to the discovery walk (D9 — degrade, never wrong answers).
    """
    if not isinstance(payload, dict):
        return None
    usable = payload.get("spec") == COVERAGE_SPEC and payload.get("encoding") == "ranges"
    return dict(payload) if usable else None


def ranges_words(envelope: dict) -> np.ndarray:
    """Shard words from a root envelope's ranges — exact expansion, or raise.

    Malformed ranges (base-crossing, wrong order, reversed endpoints) raise:
    a corrupt cache must never yield a plausible partial answer. Expansion
    is O(covered shards); containment checks on the hot path should use
    :func:`ranges_contain` instead (rank space, no materialization).
    """
    order = int(envelope["order"])
    words: list[int] = []
    for lo, hi in envelope["ranges"]:
        base = decimal_base(lo)
        lo_rank, hi_rank = decimal_rank(lo), decimal_rank(hi)
        ok = decimal_base(hi) == base and lo_rank <= hi_rank
        ok = ok and decimal_order(lo) == order and decimal_order(hi) == order
        if not ok:
            raise ValueError(f"malformed coverage range [{lo}, {hi}] at order {order}")
        words.extend(morton_word(base + rank_tail(r, order)) for r in range(lo_rank, hi_rank + 1))
    return np.unique(np.asarray(words, dtype=np.uint64))


def ranges_contain(envelope: dict, shard: str | int) -> bool:
    """Whether the envelope's ranges list one shard id — O(ranges), no expansion."""
    from moczarr.convention import morton_decimal

    decimal = morton_decimal(shard)
    if decimal_order(decimal) != int(envelope["order"]):
        return False
    base, rank = decimal_base(decimal), decimal_rank(decimal)
    return any(
        decimal_base(lo) == base and decimal_rank(lo) <= rank <= decimal_rank(hi)
        for lo, hi in envelope["ranges"]
        if decimal_base(hi) == decimal_base(lo)
    )

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


def root_coverage_and(envelope: dict, aoi) -> np.ndarray:
    """Intersection of the root ranges MOC with an AOI morton cover.

    ``aoi`` is any morton cover (mixed order allowed — mortie's ``moc_and``
    resolves containment across orders). Returns the covered shards the AOI
    touches; empty means no covered shard intersects. Expansion is
    O(covered shards) — see :func:`ranges_words`.
    """
    from mortie import moc_and

    return moc_and(ranges_words(envelope), np.asarray(aoi, dtype=np.uint64))


def box_and(coverage: dict, aoi) -> np.ndarray:
    """Intersection of a leaf envelope's tier-0 box with an AOI morton cover.

    One in-memory op on <= 4 members — the cheap AOI reject a reader runs on
    the stamp it already fetched, before paying for the bitmap sidecar. An
    empty result rejects the leaf outright (the box is a conservative
    superset: false positives possible, false negatives impossible).
    """
    from mortie import moc_and

    return moc_and(box_words(coverage), np.asarray(aoi, dtype=np.uint64))


def aoi_mask(cells, aoi) -> np.ndarray:
    """Boolean mask over ``cells``: which intersect the AOI morton cover.

    The §5 nesting predicate (prefix = ancestor), applied in BOTH
    directions: a cell is kept when it sits inside an AOI member (member
    coarser-or-equal) or contains one (member finer). This is containment,
    NOT ``np.isin`` against ``moc_and``'s output — MOC intersection returns
    a *compacted* cover (a fully-occupied subtree compacts to its parent
    word), so identity tests against it silently drop exactly the dense
    regions.

    BOTH sides may be mixed-order (issue #8, on mortie#116's per-element
    kernels): a pyramid/overview store (zagg#262) carries a parent and its
    children in one cell coordinate, and each cell resolves containment at
    its own order via ``orders_of``. The former single-order guard is gone —
    the capability replaces the constraint. Per member, the coarser-or-equal
    direction is ONE whole-array compare: ``clip2order`` is per-element
    (finer cells clip to the member's order, coarser cells pass through
    unchanged) and words at different orders can never be bit-equal (the
    order lives in the word's suffix), so pass-throughs never false-match.
    The finer direction runs once per distinct coarser cell order present
    (<= 30), comparing against the member's ancestor at that order.

    POINT-kind words (spec §1 suffix ``48..=63``) are first-class members on
    BOTH sides — the v1 posture: for containment a point counts as an
    order-29 member like any other, normalized to its order-29 area twin
    (same path; §4 — membership at a coarser level is ordinary truncation),
    and may now ride alongside area cells at ANY order (previously only
    order-29 areas kept a mixed-kind array uniform).
    """
    from mortie import clip2order, orders_of

    from moczarr.convention import point_to_area29

    cells = np.atleast_1d(np.asarray(cells, dtype=np.uint64))
    keep = np.zeros(cells.size, dtype=bool)
    if cells.size == 0:
        return keep
    # §4 normalization: containment arithmetic runs in area space (points
    # -> their order-29 twins on the same path; area words pass through).
    cells = np.asarray(point_to_area29(cells), dtype=np.uint64)
    distinct_orders = np.unique(orders_of(cells))
    members = np.atleast_1d(np.asarray(aoi, dtype=np.uint64))
    members = np.asarray(point_to_area29(members), dtype=np.uint64)
    for member, member_order in zip(members, orders_of(members)):
        keep |= np.asarray(clip2order(int(member_order), cells), dtype=np.uint64) == member
        one = np.asarray([member], dtype=np.uint64)
        for order in distinct_orders[distinct_orders < member_order]:
            keep |= cells == np.asarray(clip2order(int(order), one), dtype=np.uint64)[0]
    return keep


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

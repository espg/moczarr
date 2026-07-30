"""``zagg-composition/1`` decoding, read side — pure functions.

A composition field is one dense ``uint64`` word per cell carrying eight
8-bit lanes of quantized fractions of the cell's **signal stratum**
(``N_signal`` = the ``of`` digest's total weight — magnitude lives in the
digest, composition here). Lanes pack **LSB byte first**: lane ``i`` is bits
``8*i .. 8*i + 7``. An empty stratum packs to ``0`` (the array's fill
value).

Normative home: zagg ``docs/specification.md`` §3 — currently in review on
englacial/zagg#346 (branch ``claude/340-store-spec``; references confirmed
at merge). Narrative in zagg ``docs/signal_strata.md``; the writer is
``zagg.stats.composition``.

Quantization (§3.2) uses a **presence floor**: ``k = round(255 * c / N)``
(round-half-even), except any nonzero count quantizes to at least 1. So
``lane > 0`` means "this flag occurred" *exactly, at every N*, through
arbitrary merge chains (:func:`presence`); count recovery
``round(k * N / 255)`` is exact whenever ``N <= 254`` and within
``±(N/510 + ½)`` above (:func:`counts_from_composition`) — the writer's
quantization costs ``N/510`` and the reader's own ``round`` adds up to
another half.

Lane *meaning* is attrs-bound, never positional: the array's ``composition``
attrs block (§3.3) declares ``lanes`` (names in bit order), ``of`` (the
sibling digest field whose total weight is ``N_signal``), and ``threshold``
(the committed signal cut). :func:`unpack_composition` therefore returns
positional lanes only; name them via :func:`named_lanes`, which binds to a
:func:`parse_composition_attrs`-validated block. The spec gate is strict on
both halves — a future ``zagg-composition/2`` is adopted deliberately, never
half-parsed, and for ``/1`` the declared ``lanes`` must be *exactly* the §3.1
order (:data:`COMPOSITION_LANES`), since §3.3 permits no other. A permuted
declaration is a non-conforming store, not a relabeling instruction: binding
it would decode every lane under the wrong name.

Deliberately absent: a read-side merge. The §3.4 merge law is a write/rollup
monoid over ``(word, n_signal)`` pairs and stays zagg-owned; moczarr
consumes committed words.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

#: Spec string of the composition attrs block this module decodes.
COMPOSITION_SPEC = "zagg-composition/1"
#: Spec §3.1 lane names in bit order (LSB byte first) — five per-surface
#: marginals in ``signal_conf_ph`` column order, then the three level lanes.
#: §3.3 makes this the *exact* value a ``/1`` block declares, so
#: :func:`parse_composition_attrs` refuses any other order rather than bind a
#: store whose declaration and packed bit order disagree.
COMPOSITION_LANES = (
    "land",
    "ocean",
    "sea_ice",
    "land_ice",
    "inland_water",
    "low",
    "med",
    "high",
)
#: Lane count of a ``/1`` word (eight u8 lanes in a uint64).
COMPOSITION_LANE_COUNT = len(COMPOSITION_LANES)


def unpack_composition(words) -> np.ndarray:
    """Composition words as positional u8 lanes — ``(N,) uint64 -> (N, 8) uint8``.

    Lane ``i`` is bits ``8*i .. 8*i + 7`` of the word (LSB byte first, spec
    §3.1). Columns are POSITIONAL: what a lane *means* comes from the
    array's attrs (``composition.lanes``), never from a hardcoded order —
    use :func:`named_lanes` to bind names. Scalars pass through
    ``np.atleast_1d``; the result is always 2-D.
    """
    w = np.atleast_1d(np.asarray(words, dtype=np.uint64))
    shifts = np.arange(COMPOSITION_LANE_COUNT, dtype=np.uint64) * np.uint64(8)
    return ((w[:, None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)


def counts_from_composition(words, n_signal) -> np.ndarray:
    """Recovered per-lane counts — ``round(k * N / 255)``, ``(N, 8) int64``.

    ``n_signal`` is the per-cell total weight of the attrs-declared ``of``
    digest (scalar or ``(N,)``, broadcast against ``words``). Recovery is
    **exact** whenever ``n_signal <= 254``: the writer's quantization error is
    ``<= N/510 < 1/2`` (spec §3.2), so this function's ``round`` lands back on
    the integer count. Above that it is a bounded estimate within
    ``±(N/510 + 1/2)`` — the two roundings compose, ``N/510`` from the writer
    plus up to ``1/2`` here, i.e. ``<= floor(N/510 + 1/2)`` in integer terms —
    plus ``O(N/510)`` per re-quantizing merge. Returned, not raised, since
    presence stays exact and large-N cells are the common case. The bound
    covers lanes the writer *rounded*; a lane lifted by the presence floor
    (``k = 1`` for a count that would have quantized to 0) recovers
    ``~N/255`` by design, trading count accuracy for exact presence.
    ``n_signal <= 0`` (empty stratum) recovers all-zero counts.
    """
    lanes = unpack_composition(words).astype(np.float64)
    n = np.atleast_1d(np.asarray(n_signal, dtype=np.float64))
    if n.shape[0] not in (1, lanes.shape[0]):
        raise ValueError(f"n_signal has {n.shape[0]} cells, words has {lanes.shape[0]}")
    n = np.maximum(n, 0.0)  # n <= 0 is the empty stratum: zero counts, never negative
    return np.rint(lanes * n[:, None] / 255.0).astype(np.int64)


def presence(words) -> np.ndarray:
    """Per-lane occurrence flags — ``lane > 0``, ``(N, 8) bool``.

    Exact at **every** ``n_signal``, through arbitrary merge chains, by the
    presence floor (spec §3.2: any nonzero count quantizes to at least 1).
    Positional columns, as in :func:`unpack_composition`.
    """
    return unpack_composition(words) > 0


def parse_composition_attrs(attrs: Mapping) -> dict:
    """The validated ``composition`` block from an array's attrs — or raise.

    Strict by design, unlike the tolerant coverage caches: the word is data,
    not an index, so a reader that cannot bind it correctly must say so
    rather than degrade. Raises ``ValueError`` on a missing/malformed block,
    on any spec other than ``zagg-composition/1`` — a future revision is
    adopted deliberately, never half-parsed (§3.3 conformance rule) — and on
    a ``/1`` block whose ``lanes`` are anything but :data:`COMPOSITION_LANES`,
    which §3.3 fixes as the exact ``/1`` value.

    Returns ``{"lanes": tuple[str, ...], "of": str, "threshold": int}``:
    ``lanes`` names the bit-order lanes (LSB byte first), ``of`` names the
    sibling digest field whose total weight is ``N_signal`` (the word is
    uninterpretable without it), ``threshold`` is the committed signal cut.
    """
    block = attrs.get("composition") if isinstance(attrs, Mapping) else None
    if not isinstance(block, Mapping):
        raise ValueError("array attrs carry no composition block (spec §3.3)")
    spec = block.get("spec")
    if spec != COMPOSITION_SPEC:
        raise ValueError(
            f"unsupported composition spec {spec!r}: this reader implements "
            f"{COMPOSITION_SPEC!r} only and will not half-parse another revision"
        )
    lanes = block.get("lanes")
    if not isinstance(lanes, list | tuple) or not (
        len(lanes) == COMPOSITION_LANE_COUNT
        and all(isinstance(name, str) for name in lanes)
        and len(set(lanes)) == len(lanes)
    ):
        raise ValueError(
            f"composition.lanes must be {COMPOSITION_LANE_COUNT} unique names in bit order, "
            f"got {lanes!r}"
        )
    if tuple(lanes) != COMPOSITION_LANES:
        raise ValueError(
            f"non-conforming {COMPOSITION_SPEC} store: §3.3 fixes composition.lanes at "
            f"exactly the §3.1 order {list(COMPOSITION_LANES)}, but this array declares "
            f"{list(lanes)}. The declaration and the packed bit order disagree and the "
            f"store says nothing about which is wrong, so binding it would relabel every "
            f"lane silently"
        )
    of = block.get("of")
    if not isinstance(of, str) or not of:
        raise ValueError(f"composition.of must name the sibling digest field, got {of!r}")
    if "threshold" not in block:
        raise ValueError("composition.threshold (the committed signal cut) is required")
    return {"lanes": tuple(lanes), "of": of, "threshold": int(block["threshold"])}


def named_lanes(words, attrs: Mapping) -> dict[str, np.ndarray]:
    """Unpacked lanes keyed by their attrs-declared names — ``{name: (N,) uint8}``.

    The binding step over :func:`unpack_composition`: lane order comes from
    the validated ``composition.lanes`` attr (via
    :func:`parse_composition_attrs`), so no caller ever indexes by a
    hardcoded position and a future revision that re-means the bytes needs no
    change here. For ``/1`` that declaration is gated to the §3.1 order, so
    the mechanism is order-driven while the accepted values are not
    open-ended.
    """
    lanes = unpack_composition(words)
    names = parse_composition_attrs(attrs)["lanes"]
    return {name: lanes[:, i] for i, name in enumerate(names)}

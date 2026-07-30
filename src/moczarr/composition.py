"""``zagg-composition/1`` decoding, read side — pure functions.

A composition field is one dense ``uint64`` word per cell carrying eight
8-bit lanes of quantized fractions of the cell's **signal stratum**
(``N_signal`` = the ``of`` digest's total weight — magnitude lives in the
digest, composition here). Lanes pack **LSB byte first**: lane ``i`` is bits
``8*i .. 8*i + 7``. An empty stratum packs to ``0``.

**Precondition on the array, not on the words.** §3 says that ``0`` is also
the array's *fill value*, which is what makes an unwritten cell read as "no
flags occurred" rather than as data. Nothing in this module can check it: the
functions here take words (and attrs), never the ``zarr.Array``, so a store
declaring a nonzero ``fill_value`` for a composition field would hand every
unoccupied cell to :func:`presence` as spurious occurrences — under a
``fill_value`` of 1, a phantom ``land``. The claims below hold **provided
the array declares** ``fill_value: 0``, which §3 requires of a conforming
writer (and which the §7 conformance fixture this package tests against
does). Callers reading arrays from an untrusted writer should assert
``array.fill_value == 0`` at the open, alongside the §3.3 attrs gate.

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


def _as_words(words) -> np.ndarray:
    """``words`` as ``(N,) uint64`` — or raise rather than coerce a lie.

    §3 fixes the word as ``uint64`` and §7's conformance criterion is
    byte-exact ``uint64`` decoding with no tolerance, so the two coercions
    ``np.asarray(..., dtype=np.uint64)`` would perform silently are refused
    instead: a negative signed value wraps to "every lane at 255" (the most
    confidently wrong answer this module can give), and a float array — what
    an xarray path that promoted for a fill value hands back — truncates.
    Python ints (including above ``2**63``), lists of them, and any unsigned
    or non-negative signed array are accepted.
    """
    w = np.atleast_1d(np.asarray(words))
    if w.dtype.kind == "u":
        return w.astype(np.uint64, copy=False)
    if w.dtype.kind == "i":
        if w.size and int(w.min()) < 0:
            raise ValueError(
                f"composition words are uint64 (spec §3): {w.dtype} value {int(w.min())} "
                "is negative and would wrap to every lane at 255"
            )
        return w.astype(np.uint64)
    raise ValueError(
        f"composition words must be an integer dtype, got {w.dtype} — a uint64 word "
        "carries no NaN/fill channel, so a non-integer array can only be truncated "
        "(spec §3; §7 requires byte-exact uint64 decoding, no tolerance)"
    )


def unpack_composition(words) -> np.ndarray:
    """Composition words as positional u8 lanes — ``(N,) uint64 -> (N, 8) uint8``.

    Lane ``i`` is bits ``8*i .. 8*i + 7`` of the word (LSB byte first, spec
    §3.1). Columns are POSITIONAL: what a lane *means* comes from the
    array's attrs (``composition.lanes``), never from a hardcoded order —
    use :func:`named_lanes` to bind names. Scalars pass through
    ``np.atleast_1d``; the result is always 2-D. Raises ``ValueError`` on a
    non-integer or negative ``words`` rather than coerce it (:func:`_as_words`).
    """
    w = _as_words(words)
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
    if n.ndim != 1:
        raise ValueError(
            f"n_signal must be a scalar or 1-D per-cell weight, got shape {n.shape} — "
            "a (N, 1) column (e.g. weights.sum(axis=1, keepdims=True)) would broadcast "
            "to a 3-D result instead of (N, 8)"
        )
    if n.shape[0] not in (1, lanes.shape[0]):
        raise ValueError(f"n_signal has {n.shape[0]} cells, words has {lanes.shape[0]}")
    n = np.maximum(n, 0.0)  # n <= 0 is the empty stratum: zero counts, never negative
    return np.rint(lanes * n[:, None] / 255.0).astype(np.int64)


def presence(words) -> np.ndarray:
    """Per-lane occurrence flags — ``lane > 0``, ``(N, 8) bool``.

    Exact at **every** ``n_signal``, through arbitrary merge chains, by the
    presence floor (spec §3.2: any nonzero count quantizes to at least 1) —
    exact *for written cells*, and for unwritten ones only if the array
    declares ``fill_value: 0`` as §3 requires (see the module docstring: this
    function sees words, never the array, so it cannot check that itself).
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
    threshold = block["threshold"]
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise ValueError(
            f"composition.threshold must be an int (the committed signal cut), got "
            f"{threshold!r} — §3.3 requires each stratum digest's recorded "
            "signal_threshold to AGREE with this value, and coercing (2.7 -> 2) would "
            "fabricate that agreement"
        )
    return {"lanes": tuple(lanes), "of": of, "threshold": threshold}


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

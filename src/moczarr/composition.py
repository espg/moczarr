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
``round(k * N / 255)`` is exact whenever ``N <= 254`` and within ``±N/510``
above (:func:`counts_from_composition`).

Lane *meaning* is attrs-bound, never positional: the array's ``composition``
attrs block (§3.3) declares ``lanes`` (names in bit order), ``of`` (the
sibling digest field whose total weight is ``N_signal``), and ``threshold``
(the committed signal cut). The functions here therefore return positional
lanes only; naming them is the attrs-binding step's job (issue #20 phase 2).

Deliberately absent: a read-side merge. The §3.4 merge law is a write/rollup
monoid over ``(word, n_signal)`` pairs and stays zagg-owned; moczarr
consumes committed words.
"""

from __future__ import annotations

import numpy as np

#: Spec string of the composition attrs block this module decodes.
COMPOSITION_SPEC = "zagg-composition/1"
#: Lane count of a ``/1`` word (eight u8 lanes in a uint64).
COMPOSITION_LANE_COUNT = 8


def unpack_composition(words) -> np.ndarray:
    """Composition words as positional u8 lanes — ``(N,) uint64 -> (N, 8) uint8``.

    Lane ``i`` is bits ``8*i .. 8*i + 7`` of the word (LSB byte first, spec
    §3.1). Columns are POSITIONAL: what a lane *means* comes from the
    array's attrs (``composition.lanes``), never from a hardcoded order.
    Scalars pass through ``np.atleast_1d``; the result is always 2-D.
    """
    w = np.atleast_1d(np.asarray(words, dtype=np.uint64))
    shifts = np.arange(COMPOSITION_LANE_COUNT, dtype=np.uint64) * np.uint64(8)
    return ((w[:, None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)


def counts_from_composition(words, n_signal) -> np.ndarray:
    """Recovered per-lane counts — ``round(k * N / 255)``, ``(N, 8) int64``.

    ``n_signal`` is the per-cell total weight of the attrs-declared ``of``
    digest (scalar or ``(N,)``, broadcast against ``words``). Recovery is
    **exact** whenever ``n_signal <= 254`` (quantization error
    ``<= N/510 < 1/2``, spec §3.2); above that it is a bounded estimate
    within ``±N/510`` (plus ``O(N/510)`` per re-quantizing merge) — returned,
    not raised, since presence stays exact and large-N cells are the common
    case. ``n_signal <= 0`` (empty stratum) recovers all-zero counts.
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

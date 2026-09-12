"""Native t-digest algebra — pure numpy, the reader's own kernels (issue #64).

The stored payload is the whole byte contract (zagg spec §2.1): a populated
cell decodes to a ``(k, 2)`` float32 centroid array — column 0 the mean,
column 1 the weight — rows ascending by mean, and an absent cell is the
zero-length ``(0, 2)`` array. The *algebra* over those bytes (Dunning's k1
scale, the ``delta`` budget, merge order) is deliberately **not** in the spec
(§2.3, informative): until this module it lived only in ``zagg.stats.tdigest``,
which made the ``moczarr[zagg]`` extra load-bearing for every rasterization
and surface read. These kernels implement the same algebra natively so the
reader decodes from the spec + fixtures alone; equivalence with zagg is
pinned by the parity gate (``tests/test_tdigest.py``, run under the demoted
``moczarr[zagg]`` extra), exact value equality — not ``np.isclose``.

Two standing traps, honoured here and required of every caller:

- **Sort before interp.** Every evaluation kernel walks centroids in mean
  order (spec §2.1's stored order). A concatenation of digests is a stack of
  ascending runs, not one ascending run, and evaluating it unsorted is a
  *silent* wrong answer (an observed 0.62 CDF error on a [0, 1] axis). A
  stored digest arrives sorted; anything assembled from more than one MUST be
  re-sorted by ``(mean, weight)`` first — :func:`merge_tdigests_kway` does its
  own lexsort, the evaluation kernels do not.
- **Rank positions are midpoints of weight intervals, not the raw cumsum.**
  A centroid of weight ``w`` sits at cumulative weight ``cum_before + w/2``
  (:func:`cdf_from_tdigest`); using ``cumsum`` directly shifts every
  position by half a centroid.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cdf_from_tdigest",
    "merge_tdigests_kway",
    "quantile_from_tdigest",
]

#: The compression budget every shipped zagg store is built with
#: (``zagg.stats.tdigest._DEFAULT_DELTA``); informative per spec §2.3 — a
#: reader never needs it to decode, only to fold digests it merges itself.
DEFAULT_DELTA = 512


def _k1_scale(q: np.ndarray, delta: float) -> np.ndarray:
    """Dunning's k1 scale ``k(q) = delta * (arcsin(2q - 1)/pi + 1/2)``.

    Maps each cumulative rank fraction ``q ∈ [0, 1]`` onto ``[0, delta]``,
    steepest at the tails — bounding each centroid to span ≤ 1 unit of k is
    what buys the t-digest's tight-tails accuracy profile (spec §2.3,
    informative).
    """
    qc = np.clip(q, 0.0, 1.0)
    return delta * (np.arcsin(2.0 * qc - 1.0) / np.pi + 0.5)


def _compress(
    means: np.ndarray, weights: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """k1-bounded greedy compression of mean-ascending weighted sub-centroids.

    ``means`` must be ascending and ``weights`` strictly positive, both 1-D
    float64. Sub-centroid ``i`` joins the open centroid while the k1 value at
    its right edge stays within 1.0 of the centroid's left edge; the first
    breaker opens the next centroid (one ``searchsorted`` per output
    centroid, so the loop runs ~delta times regardless of input length).
    Output means are the weighted averages of the joined runs.

    This is the same greedy rule as zagg's ``_compress`` — same k1 values at
    the same rank fractions, same join test — because the parity gate demands
    exact value equality, float associativity included (the rank denominator
    is ``cumw[-1]``, the join test is ``searchsorted`` on ``k_left + 1.0``,
    the means one weighted ``sum / weight`` per centroid).
    """
    n = len(means)
    cumw = np.cumsum(weights)
    total = float(cumw[-1])
    # k1 at each sub-centroid's right edge; monotonic in i since weights > 0.
    k_right = _k1_scale(cumw / total, delta)
    k_left0 = float(_k1_scale(np.zeros(1), delta)[0])

    starts_list: list[int] = []
    s = 0
    while s < n:
        starts_list.append(s)
        k_left = k_left0 if s == 0 else float(k_right[s - 1])
        e = int(np.searchsorted(k_right, k_left + 1.0, side="right"))
        if e <= s:
            # The next sub-centroid alone overflows the k-span (steep tail):
            # the centroid holds only its start. Guarantees forward progress.
            e = s + 1
        s = e

    starts = np.asarray(starts_list, dtype=np.int64)
    out_weights = np.add.reduceat(weights, starts)
    out_means = np.add.reduceat(means * weights, starts) / out_weights
    return out_means, out_weights


def merge_tdigests_kway(digests: list[np.ndarray], delta: int = DEFAULT_DELTA) -> np.ndarray:
    """Merge many t-digests in one flat pass — **order-independent**.

    Concatenates every centroid array, sorts once by ``(mean, weight)``, and
    re-compresses under the k1 budget: a permutation of the inputs returns
    the same digest, byte for byte, because the sorted concatenation is
    intrinsic to the multiset of centroids (the weight tie-key canonicalizes
    centroids sharing a mean, which a stable mean-only sort would leave in
    input order). That is permutation-independence of one flat k-way call —
    t-digest merging is otherwise order-dependent (spec §2.3, informative),
    which is why a reader folding strata MUST use this law and never a
    pairwise left-fold.

    Weight semantics: merging is legal only between payloads carrying the
    same ``weights`` declaration (spec §2.0 — counts with counts, flux with
    flux); this kernel sees bare centroid arrays, so that gate is the
    caller's.

    Companion channels (the located / temporal word vectors, spec §2.2/§8.3)
    are writer-side folds and stay zagg-owned; this reader kernel merges
    digests only.

    Parameters
    ----------
    digests : list of ndarray
        Centroid arrays ``(k, 2)`` per spec §2.1. Empty digests are skipped.
    delta : int, optional
        Compression budget (default :data:`DEFAULT_DELTA`, every shipped
        store's build budget).

    Returns
    -------
    ndarray, shape (k_merged, 2), dtype float32
        Merged, re-compressed centroid array; ``(0, 2)`` when every input is
        empty. A single non-empty input is returned as-is (float32 copy, no
        re-compression — already a valid digest).
    """
    arrs = [np.asarray(d, dtype=np.float64) for d in digests]
    arrs = [d for d in arrs if d.size]
    if not arrs:
        return np.empty((0, 2), dtype=np.float32)
    if len(arrs) == 1:
        return arrs[0].astype(np.float32)

    combined = np.concatenate(arrs, axis=0)
    # lexsort, least-significant key first: mean primary, weight secondary.
    order = np.lexsort((combined[:, 1], combined[:, 0]))
    combined = combined[order]
    out_means, out_weights = _compress(combined[:, 0], combined[:, 1], float(delta))
    out = np.empty((len(out_means), 2), dtype=np.float32)
    out[:, 0] = out_means
    out[:, 1] = out_weights
    return out


def quantile_from_tdigest(digest: np.ndarray, q: float) -> float:
    """Estimate quantile ``q`` from a mean-ascending centroid array.

    The standard t-digest interpolation: the target rank is ``q * (n - 1)``
    (0-indexed observations, total weight ``n``); centroid ``k`` of weight
    ``w_k`` owns the rank interval ``[upper_{k-1}, upper_k - 1]`` (``upper``
    the weight cumsum), and the value interpolates linearly between the
    midpoints of adjacent centroid means — the digest's piecewise-linear
    inverse CDF, clamped to ``means[0]`` / ``means[-1]`` at the ends. A
    single-observation interval returns its centroid mean exactly.

    ``digest`` must satisfy spec §2.1's stored order (ascending by mean); a
    concatenation must be re-sorted first (module docstring, trap 1).

    Parameters
    ----------
    digest : ndarray, shape (k, 2)
        Centroid array per spec §2.1.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    float
        Approximate quantile value; NaN for an empty digest.
    """
    if len(digest) == 0:
        return float("nan")
    means = np.asarray(digest[:, 0], dtype=np.float64)
    weights = np.asarray(digest[:, 1], dtype=np.float64)
    n = weights.sum()
    upper = np.cumsum(weights)
    target = q * (n - 1)
    for k in range(len(means)):
        lo = 0.0 if k == 0 else upper[k - 1]
        hi = upper[k] - 1.0
        if target <= hi:
            if hi <= lo:
                return float(means[k])
            frac = (target - lo) / (hi - lo)
            if k == 0:
                lo_val = means[0]
                hi_val = means[0] if len(means) == 1 else (means[0] + means[1]) / 2.0
            else:
                lo_val = (means[k - 1] + means[k]) / 2.0
                hi_val = means[k] if k == len(means) - 1 else (means[k] + means[k + 1]) / 2.0
            return float(lo_val + frac * (hi_val - lo_val))
    return float(means[-1])


def cdf_from_tdigest(digest: np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
    """Cumulative weight at value(s) ``x`` — the value→rank inverse.

    Each centroid sits at the **midpoint of its weight interval**,
    cumulative weight ``cum_before + w/2`` (module docstring, trap 2); the
    CDF interpolates cumulative weight linearly in value-space between
    adjacent centroid means and clamps flat outside ``[means[0],
    means[-1]]`` — monotonic non-decreasing from ``0`` to the total weight.
    A single-centroid digest is a step at its mean. This is the primitive
    that fills evenly-spaced value bins from a digest
    (:func:`moczarr.hhdc.rasterize_cell`).

    ``digest`` must satisfy spec §2.1's stored order (ascending by mean); a
    concatenation must be re-sorted first (module docstring, trap 1).

    Parameters
    ----------
    digest : ndarray, shape (k, 2)
        Centroid array per spec §2.1.
    x : float or ndarray
        Value(s) at which to evaluate.

    Returns
    -------
    float or ndarray
        Cumulative weight in ``[0, total_weight]`` — a scalar for scalar
        ``x``, float64 array otherwise; NaN (shaped like ``x``) for an
        empty digest.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    scalar = x_arr.ndim == 0

    if len(digest) == 0:
        out = np.full(x_arr.shape, np.nan, dtype=np.float64)
        return float(out) if scalar else out

    means = np.asarray(digest[:, 0], dtype=np.float64)
    weights = np.asarray(digest[:, 1], dtype=np.float64)
    total = float(weights.sum())

    cum_upper = np.cumsum(weights)
    cum_center = cum_upper - weights / 2.0

    if len(means) == 1:
        out = np.where(x_arr >= means[0], total, 0.0)
        return float(out) if scalar else out.astype(np.float64)

    # np.interp clamps to the endpoint values outside [means[0], means[-1]],
    # giving the flat 0 / total tails.
    out = np.interp(x_arr, means, cum_center, left=0.0, right=total)
    return float(out) if scalar else out

"""Exact NESTED ``cell_ids`` fabricated from packed morton words.

The morton-only storage decision (englacial/zagg#262: "NESTED is fabricated,
never stored") drops the stored NESTED ``cell_ids`` array from hive leaves;
``morton`` becomes the only stored cell coordinate. This module is the
load-bearing replacement — the ratified gate for zagg's writer flip: NESTED
ids are derived **exactly** from the packed words via mortie's vectorized
``mort2healpix``, so Python consumers keep NESTED for free.

Order-24 seam (zagg#262, open): NESTED ids above order 24 exceed the
float64-exact integer range (``2**53``), so JS/browser consumers cannot hold
them losslessly as Numbers. The ratified 29→24 clip policy for
unknown-resolution-at-29 words lands here once the resolution discriminator
metadata is pinned on the zagg#262 thread; until then fabrication above
order 24 only warns (see :data:`FLOAT64_EXACT_MAX_ORDER`).
"""

from __future__ import annotations

import warnings

import numpy as np

#: Highest HEALPix order whose NESTED ids all stay below ``2**53`` and are
#: therefore float64-exact (safe as JS Numbers in browser consumers).
FLOAT64_EXACT_MAX_ORDER = 24


def _warn_above_float64_exact(order: int) -> None:
    warnings.warn(
        f"fabricating NESTED cell_ids at order {order}: ids above order "
        f"{FLOAT64_EXACT_MAX_ORDER} exceed the float64-exact integer range "
        f"(2**53) and are unsafe as JS Numbers in browser consumers; the "
        f"29->24 clip policy lands with the zagg#262 resolution discriminator",
        UserWarning,
        stacklevel=3,
    )


def fabricate_cell_ids(morton_words, *, level: int | None = None) -> np.ndarray:
    """Exact NESTED ``uint64`` cell ids of packed morton words.

    Pure function (mortie/numpy only): ``mort2healpix`` on the packed words.
    The words must share one order (mortie rejects mixed orders — fine for
    hive stores, whose cell coordinates are single-order at the manifest's
    ``cell_order``).

    Parameters
    ----------
    morton_words : array-like
        Packed ``uint64`` morton words (the ``morton`` coordinate).
    level : int, optional
        Expected HEALPix order of the words. When given it is cross-checked
        against the order mortie derives from the words themselves and a
        mismatch raises ``ValueError``.

    Returns
    -------
    numpy.ndarray
        NESTED cell ids, ``uint64``, same length as the input. Fabricating
        above order :data:`FLOAT64_EXACT_MAX_ORDER` emits a ``UserWarning``
        (float64/browser limitation; clip policy pending on zagg#262).
    """
    words = np.asarray(morton_words, dtype=np.uint64).ravel()
    if words.size == 0:
        if level is not None and level > FLOAT64_EXACT_MAX_ORDER:
            _warn_above_float64_exact(level)
        return np.empty(0, dtype=np.uint64)
    from mortie import mort2healpix

    ids, order = mort2healpix(words)
    if level is not None and level != order:
        raise ValueError(f"level={level} does not match the words' order {order}")
    if order > FLOAT64_EXACT_MAX_ORDER:
        _warn_above_float64_exact(order)
    return np.asarray(ids, dtype=np.uint64)

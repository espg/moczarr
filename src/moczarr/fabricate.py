"""Exact NESTED ``cell_ids`` fabricated from packed morton words.

The morton-only storage decision (englacial/zagg#262: "NESTED is fabricated,
never stored") drops the stored NESTED ``cell_ids`` array from hive leaves;
``morton`` becomes the only stored cell coordinate. This module is the
load-bearing replacement — the ratified gate for zagg's writer flip: NESTED
ids are derived **exactly** from the packed words via mortie's vectorized
``mort2healpix``, so Python consumers keep NESTED for free.

Order-24 seam (resolved per the final mortie spec §1/§4 — kind is carried
by the word encoding): NESTED ids above order 24 exceed the float64-exact
integer range (``2**53``), so JS/browser consumers cannot hold them
losslessly as Numbers. POINT-kind words (§1 suffix ``48..=63`` — an
order-29 location with no area claim) clip to order 24 for the NESTED view:
membership at a coarser level is ordinary truncation (§4), so nothing
labelled changes. Genuine AREA cells are NEVER clipped — coarsening an area
cell changes the labelled thing (that is aggregation, zagg D24) — so
fabrication of area cells above order 24 keeps the exact ids and warns
(see :data:`FLOAT64_EXACT_MAX_ORDER`).
"""

from __future__ import annotations

import warnings

import numpy as np

#: Highest HEALPix order whose NESTED ids all stay below ``2**53`` and are
#: therefore float64-exact (safe as JS Numbers in browser consumers).
FLOAT64_EXACT_MAX_ORDER = 24


def _warn_above_float64_exact(order: int, stacklevel: int) -> None:
    warnings.warn(
        f"fabricating NESTED cell_ids at order {order}: ids above order "
        f"{FLOAT64_EXACT_MAX_ORDER} exceed the float64-exact integer range "
        f"(2**53) and are unsafe as JS Numbers in browser consumers. Area "
        f"cells keep their exact order by design (spec §4: coarsening an "
        f"area cell changes the labelled thing) — only point-kind words clip",
        UserWarning,
        stacklevel=stacklevel,
    )


def fabricate_cell_ids(
    morton_words, *, level: int | None = None, _stacklevel: int = 3
) -> np.ndarray:
    """Exact NESTED ``uint64`` cell ids of packed morton words.

    Pure function (mortie/numpy only): ``mort2healpix`` on the packed words.
    The AREA words must share one order (mortie rejects mixed orders — fine
    for hive stores, whose cell coordinates are single-order at the
    manifest's ``cell_order``). POINT-kind words (spec §1 suffix
    ``48..=63``) may ride alongside order-29 areas — mixed kinds are
    well-formed per §4 — and clip to order
    :data:`FLOAT64_EXACT_MAX_ORDER` for the NESTED view: a point has no
    area claim, so coarser membership is ordinary truncation (§4), keeping
    every fabricated id float64-exact. Area cells NEVER clip. Because
    mortie's kernel takes one order per call, points (clipped to 24) and
    areas fabricate separately and reassemble by position.

    Parameters
    ----------
    morton_words : array-like
        Packed ``uint64`` morton words (the ``morton`` coordinate).
    level : int, optional
        Expected HEALPix order of the words' ENCODING. When given it is
        cross-checked against the order mortie derives from the area words
        (and against 29 for point words — their encoded order, not the
        clipped NESTED order) and a mismatch raises ``ValueError``.
    _stacklevel : int, optional
        Frame depth for the above-order-24 ``UserWarning`` so it lands on
        user code. Defaults to ``3`` (a direct call); the ``open_hive`` path
        passes ``4`` to account for the extra opener frame. Internal.

    Returns
    -------
    numpy.ndarray
        NESTED cell ids, ``uint64``, same length as the input. Fabricating
        AREA cells above order :data:`FLOAT64_EXACT_MAX_ORDER` emits a
        ``UserWarning`` (float64/browser limitation; §4 — area cells keep
        their exact order). Point words fabricate silently (already
        clipped to the float64-exact ceiling).
    """
    words = np.asarray(morton_words, dtype=np.uint64).ravel()
    if words.size == 0:
        if level is not None and level > FLOAT64_EXACT_MAX_ORDER:
            _warn_above_float64_exact(level, _stacklevel)
        return np.empty(0, dtype=np.uint64)
    from mortie import mort2healpix

    from moczarr.convention import is_point_word, point_to_area29

    point_mask = np.asarray(is_point_word(words))
    if not point_mask.any():
        ids, order = mort2healpix(words)
        if level is not None and level != order:
            raise ValueError(f"level={level} does not match the words' order {order}")
        if order > FLOAT64_EXACT_MAX_ORDER:
            _warn_above_float64_exact(order, _stacklevel)
        return np.asarray(ids, dtype=np.uint64)
    if level is not None and level != 29:
        raise ValueError(f"level={level} does not match the point words' encoded order 29")
    from mortie import clip2order

    ids = np.empty(words.size, dtype=np.uint64)
    clipped = clip2order(FLOAT64_EXACT_MAX_ORDER, point_to_area29(words[point_mask]))
    point_ids, _order24 = mort2healpix(np.asarray(clipped, dtype=np.uint64))
    ids[point_mask] = np.asarray(point_ids, dtype=np.uint64)
    areas = words[~point_mask]
    if areas.size:
        area_ids, area_order = mort2healpix(areas)
        if level is not None and level != area_order:
            raise ValueError(f"level={level} does not match the area words' order {area_order}")
        if area_order > FLOAT64_EXACT_MAX_ORDER:
            _warn_above_float64_exact(area_order, _stacklevel)
        ids[~point_mask] = np.asarray(area_ids, dtype=np.uint64)
    return ids

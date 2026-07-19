"""Cross-resolution truncation join (phase 6, espg/moczarr#1; zagg#198's O4).

The §5 nesting predicate — ``fine_id.startswith(coarse_id)`` on decimal
strings, ``clip2order(coarse_level, fine_words) == coarse_word`` on packed
words — makes cross-resolution work a *lookup*, not machinery:
:func:`parent_cells` fabricates the parent coordinate (one vectorized mortie
call) and :func:`join_coarse` broadcasts coarse variables onto the fine
domain by looking up each fine cell's parent row. The aggregation direction
is deliberately NOT here: fine→coarse reduction is plain xarray groupby over
the parent coordinate::

    parents = moczarr.parent_cells(fine, level)
    coarse = fine.assign_coords(parents).groupby(parents.name).mean()

and *persisted* coarse products are zagg#300's pyramid sweep.

Origin-agnostic by design: the two datasets may come from two stores or
(once englacial/zagg#299 lands) two products of one multi-product store —
everything here operates on coordinate values, never on paths or leaf
basenames (tracking: espg/moczarr#11).
"""

from __future__ import annotations

import numpy as np


def parent_cells(obj, level: int):
    """Parent morton cells at ``level`` — the vectorized truncation.

    Parameters
    ----------
    obj : xarray.Dataset or array-like
        A dataset carrying a ``morton`` coordinate (an ``open_hive`` result;
        on a moc-indexed dataset the coordinate is fabricated arithmetic —
        no store reads), or a packed ``uint64`` word array.
    level : int
        HEALPix order to truncate to. Must be at or above (coarser than or
        equal to) the cells' own order: ``level`` equal to the cells' order
        is the identity (``clip2order`` returns words at or below the target
        unchanged), and a finer ``level`` raises — truncation cannot refine.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        For a Dataset: a ``uint64`` DataArray named ``parent_o{level}`` on
        the cells dimension, ready for ``assign_coords``/``groupby``. For an
        array: the parent words as a ``uint64`` array.
    """
    import xarray as xr  # lazy: the package root stays xarray-import-free

    level = int(level)
    if not 0 <= level <= 29:
        raise ValueError(f"level {level} outside the packed morton range 0..29")
    is_dataset = isinstance(obj, xr.Dataset)
    if is_dataset:
        if "morton" not in obj.variables:
            raise ValueError(
                "dataset has no 'morton' coordinate — parent_cells needs a morton "
                "dataset (an open_hive result) or a packed word array"
            )
        source = obj["morton"]
        words = np.asarray(source.values, dtype=np.uint64)
    else:
        words = np.asarray(obj, dtype=np.uint64).ravel()
    if words.size == 0:
        raise ValueError("cannot derive an order from zero cells")
    from mortie import clip2order, infer_order_from_morton

    order = int(infer_order_from_morton(words))
    if level > order:
        raise ValueError(
            f"level {level} is finer than the cells' order {order}; parent_cells "
            f"truncates toward coarser levels only (level <= {order})"
        )
    parents = np.asarray(clip2order(level, words), dtype=np.uint64)
    if is_dataset:
        return xr.DataArray(parents, dims=source.dims, name=f"parent_o{level}")
    return parents

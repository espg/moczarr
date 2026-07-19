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

from collections.abc import Callable

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


def _coarse_lookup(coarse) -> tuple[int, Callable]:
    """``(level, parents -> positions)`` for the coarse dataset's morton domain.

    The level derivation: a moc-indexed coarse answers from its interval set
    (``MortonMocIndex.level`` — and ``rank`` is exactly the phase-6b repeated-
    label lookup, no fabrication); any other morton dataset answers from the
    words themselves (``infer_order_from_morton`` on the coordinate values,
    positions via a pandas ``get_indexer``). Missing parents map to ``-1``.
    """
    import xarray as xr

    from moczarr.moc_index import MortonMocIndex

    if not isinstance(coarse, xr.Dataset) or "morton" not in coarse.variables:
        raise ValueError(
            "coarse has no 'morton' coordinate — join_coarse needs a morton "
            "dataset (an open_hive result, or a groupby product carrying the "
            "parent words as its 'morton' coordinate)"
        )
    index = dict(coarse.xindexes).get("morton")
    if isinstance(index, MortonMocIndex):
        return index.level, lambda parents: index.ranges.rank(parents, missing=-1)
    import pandas as pd
    from mortie import infer_order_from_morton

    words = np.asarray(coarse["morton"].values, dtype=np.uint64)
    if words.size == 0:
        raise ValueError("coarse has zero cells — nothing to join")
    level = int(infer_order_from_morton(words))
    labels = pd.Index(words)
    return level, lambda parents: labels.get_indexer(parents)


def join_coarse(fine, coarse, *, variables=None, how="left", suffix=None):
    """Broadcast coarse variables onto the fine domain — the §5 lookup join.

    Each fine cell's parent at the coarse dataset's level (derived from the
    coarse ``morton`` coordinate/index — see :func:`_coarse_lookup`) selects
    one coarse row; the selected variables land on fine's cells dimension,
    answering "for this fine observation, what does the containing coarse
    cell say." A values-level operation: any pairing of index kinds (pandas,
    moc, or no index at all) on either side joins identically.

    Parameters
    ----------
    fine, coarse : xarray.Dataset
        Morton datasets (``open_hive`` results, or anything carrying a
        ``morton`` coordinate of packed words — e.g. a groupby product).
        ``coarse`` must sit at a level at or above fine's (equal level is an
        identity lookup); a finer ``coarse`` raises via :func:`parent_cells`.
    variables : sequence of str, optional
        Which coarse data variables to join; default all of them.
    how : {"left", "inner"}
        ``"left"`` (default) keeps every fine row; rows whose parent is
        absent from coarse get fill values with xarray's missing-value
        semantics (the ``reindex``/``where`` defaults): float variables fill
        with NaN, integer variables are **promoted to float64** and fill
        with NaN, datetimes fill with NaT. No dtype magic beyond that —
        ``how="inner"`` drops the uncovered fine rows instead and preserves
        integer dtypes.
    suffix : str, optional
        Renames every joined variable (``name + suffix``). Collisions with a
        fine variable are never resolved silently: they raise, naming the
        colliding variables and suggesting ``suffix="_o{level}"``.

    Returns
    -------
    xarray.Dataset
        ``fine`` (all rows, or the covered rows for ``"inner"``) with the
        joined coarse variables added on its cells dimension. A moc-indexed
        fine keeps its lazy index (``"inner"`` subsets are monotonic).
    """
    import xarray as xr

    if how not in ("left", "inner"):
        raise ValueError(f"how={how!r}: expected 'left' or 'inner'")
    level, lookup = _coarse_lookup(coarse)
    parents = parent_cells(fine, level)  # validates fine + the level direction
    positions = np.asarray(lookup(parents.values), dtype=np.int64)

    names = list(coarse.data_vars) if variables is None else list(variables)
    unknown = [name for name in names if name not in coarse.data_vars]
    if unknown:
        raise ValueError(f"variables not in coarse: {', '.join(sorted(unknown))}")
    renamed = {name: f"{name}{suffix}" if suffix else name for name in names}
    collisions = [final for final in renamed.values() if final in fine.variables]
    if collisions:
        raise ValueError(
            f"joined variables collide with fine variables: "
            f"{', '.join(sorted(collisions))}; pass suffix= to rename them "
            f"(e.g. suffix='_o{level}')"
        )

    fdim = fine["morton"].dims[0]
    cdim = coarse["morton"].dims[0]
    mask = positions >= 0
    if how == "inner":
        keep = np.flatnonzero(mask)
        fine = fine.isel({fdim: keep})  # monotonic: a moc index survives
        take, fill_mask = positions[mask], None
    else:
        # Missing parents point at row 0 for the take, then blank out below.
        take, fill_mask = np.where(mask, positions, 0), mask
    joined = coarse[names]
    joined = joined.drop_vars(list(joined.coords)).isel({cdim: take})
    if cdim in joined.dims and cdim != fdim:
        joined = joined.rename({cdim: fdim})
    if fill_mask is not None and not fill_mask.all():
        covered = xr.DataArray(fill_mask, dims=fdim)
        for name in joined.data_vars:
            if fdim in joined[name].dims:
                joined[name] = joined[name].where(covered)
    joined = joined.rename(renamed)
    return fine.assign({name: joined[name] for name in joined.data_vars})

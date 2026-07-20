"""``MortonMocIndex``: the MOC-backed lazy xarray index (core, xarray-only).

The §6 "domain = MOC, dense views fabricated lazily" centerpiece (phase 5,
espg/moczarr#1): an :class:`xarray.Index` whose state is a
:class:`~moczarr.ranges.MortonRanges` interval set instead of a materialized
cell array — the ``HealpixMocIndex`` analog (xarray-contrib/xdggs#143) on
the numpy interval substrate. Selection, subsetting, and alignment are rank
arithmetic; the coordinate itself is fabricated on demand
(:meth:`MortonMocIndex.create_variables`), so an ``open_hive`` result can
carry a fully functional cell index without ever reading a coordinate chunk.

Core placement is deliberate: this module imports xarray only (no xdggs).
The accessor/registry wiring lives in ``moczarr.dggs`` (the ``[xdggs]``
extra), which wraps this index behind ``index_kind="moc"`` the way upstream
``HealpixIndex`` wraps ``HealpixMocIndex``.

Mixing with the pandas-backed index raises pointedly: an interval set and a
hash table have no shared alignment currency, so both sides of an
align/join must be opened with the same ``index_kind``.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from xarray.core.indexing import IndexSelResult

from moczarr.convention import morton_word
from moczarr.ranges import MortonRanges


def _warn_index_dropped() -> None:
    """Signal that an unrepresentable selection dropped the lazy index.

    An interval set has no reordering freedom, so a non-monotonic positional
    pick cannot round-trip through it. The selected *values* stay correct
    (xarray falls back to the already-fabricated coordinate), but the result
    carries no ``MortonMocIndex`` — a silent type change the reviewer asked
    to make observable. Both ``sel`` (which xarray lowers to a positional
    ``isel``) and a direct ``isel`` land here, so the single warning site
    lives on the ``isel`` drop path — warning in ``sel`` too would
    double-fire the same user action. Picks containing *duplicates* do NOT
    warn (phase 6b): a repeated-label lookup is the sanctioned truncation-join
    pattern (``coarse.sel(morton=parent_cells(fine, level))``), never
    representable by any interval index, so its silent drop is by design, not
    an accident worth surfacing. ``stacklevel=3`` aims the warning past this
    helper and ``isel`` toward the caller; xarray's variable internal depth
    makes an exact user frame unreachable by a fixed level.
    """
    warnings.warn(
        "MortonMocIndex: this positional selection is not representable as an "
        "interval set (a non-monotonic pick); the lazy index was dropped. "
        "Selected values remain correct, but the result carries no morton "
        'index (reopen with index_kind="moc" or set_xindex to restore it).',
        UserWarning,
        stacklevel=3,
    )


def _normalize_chunks(chunks: int | tuple[int, ...] | None, size: int) -> tuple[int, ...] | None:
    """Chunk sizes covering ``size``, or ``None`` for the plain-numpy path."""
    if chunks is None:
        return None
    if isinstance(chunks, int):
        if chunks <= 0:
            raise ValueError(f"chunks must be positive (got {chunks})")
        full, rest = divmod(size, chunks)
        return (chunks,) * full + ((rest,) if rest else ())
    chunks = tuple(int(c) for c in chunks)
    if sum(chunks) != size:
        raise ValueError(f"chunks {chunks} do not sum to the domain size {size}")
    return chunks


def _construct_chunk_ranges(chunks: tuple[int, ...], until: int) -> Iterator[tuple[int, slice]]:
    """``(chunksize, positional slice)`` per chunk (the xdggs helper shape)."""
    start = 0
    for chunksize in chunks:
        stop = min(start + chunksize, until)
        yield stop - start, slice(start, stop)
        start = stop


def _extract_chunk(ranges: MortonRanges, slice_: slice) -> np.ndarray:
    """Fabricate one positional chunk of the domain (the dask task body)."""
    return ranges.subset(slice_).fabricate()


class MortonMocIndex(xr.Index):
    """Lazy cell index over a :class:`~moczarr.ranges.MortonRanges` domain.

    ``sel`` accepts packed ``uint64`` words, decimal strings, or lists/arrays
    of either; ``isel`` and alignment (``join``/``reindex_like``) run in
    interval algebra. Selections an interval set cannot represent (scalar
    collapse, non-monotonic positional picks) degrade by dropping the lazy
    index — xarray then works on the already-fabricated coordinate — never
    by materializing a second truth.
    """

    def __init__(
        self,
        ranges: MortonRanges,
        *,
        dim: str,
        name: str = "morton",
        chunks: tuple[int, ...] | None = None,
    ):
        self._ranges = ranges
        self._dim = dim
        self._name = name
        self._chunks = chunks

    # -- construction -----------------------------------------------------

    @classmethod
    def from_moc(
        cls,
        source,
        *,
        cell_order: int,
        aoi=None,
        dim: str = "cells",
        name: str = "morton",
        chunks: int | tuple[int, ...] | None = None,
    ) -> "MortonMocIndex":
        """The first-class constructor: a domain from coverage arithmetic.

        ``source`` is a root coverage envelope (the ``"ranges"`` dict a hive
        store's ``coverage.moc`` holds), an array of shard words (the walk
        fallback), or an existing :class:`MortonRanges`. ``aoi`` (a morton
        cover, mixed orders allowed) intersects the domain in interval
        space. ``chunks`` requests dask-backed coordinate fabrication (an
        int or explicit chunk sizes); the default fabricates plain numpy.
        """
        if isinstance(source, MortonRanges):
            ranges = source
            if ranges.cell_order != int(cell_order):
                raise ValueError(
                    f"ranges at order {ranges.cell_order} do not match cell_order {cell_order}"
                )
        elif isinstance(source, dict):
            ranges = MortonRanges.from_root_coverage(source, cell_order)
        else:
            ranges = MortonRanges.from_shards(source, cell_order)
        if aoi is not None:
            ranges = ranges.intersect(aoi)
        return cls(ranges, dim=dim, name=name, chunks=_normalize_chunks(chunks, ranges.size))

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, xr.Variable], *, options: Mapping[str, Any]
    ) -> "MortonMocIndex":
        """Build from a materialized ``morton`` coordinate (``ds.set_xindex``).

        The materialized→lazy direction: run-detects the words into
        intervals. ``level`` comes from ``options`` or the variable's
        ``level``/``cell_order`` attrs, else the words' own order.
        """
        if len(variables) != 1:
            raise ValueError("MortonMocIndex indexes exactly one coordinate")
        name, var = next(iter(variables.items()))
        if var.ndim != 1:
            raise ValueError("only 1D cell ids are supported")
        options = dict(options)
        level = options.pop("level", None)
        if options:
            raise ValueError(f"unknown options {sorted(options)} (only 'level' is accepted)")
        if level is None:
            attrs = var.attrs
            level = attrs.get("level", attrs.get("cell_order"))
        words = np.asarray(var.data, dtype=np.uint64)
        ranges = MortonRanges.from_cell_words(words, None if level is None else int(level))
        return cls(ranges, dim=var.dims[0], name=name)

    def _replace(self, ranges: MortonRanges) -> "MortonMocIndex":
        # Positional chunk sizes do not survive a subset; the replacement
        # falls back to plain-numpy fabrication (chunks are an open-time
        # optimization, not index state worth re-deriving).
        return type(self)(ranges, dim=self._dim, name=self._name)

    # -- introspection ----------------------------------------------------

    @property
    def size(self) -> int:
        """Number of cells in the indexed domain."""
        return self._ranges.size

    @property
    def level(self) -> int:
        """HEALPix order of the domain's cells (manifest ``cell_order``)."""
        return self._ranges.cell_order

    @property
    def ranges(self) -> MortonRanges:
        """The interval set backing this index."""
        return self._ranges

    # -- xarray Index API -------------------------------------------------

    @classmethod
    def concat(
        cls,
        indexes: Sequence["MortonMocIndex"],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> "MortonMocIndex":
        """Concatenate disjoint, ascending MOC domains (the batch-sweep case).

        ``xr.concat`` of moc-indexed datasets is supported when the domains
        are already disjoint and in ascending word order end to end — the
        AOI-tile / batch-sweep pattern, where each open covers a distinct
        spatial block. The data variables concatenate in the given block
        order, and the fabricated ``morton`` coordinate has to match row for
        row; that holds exactly when the concatenated interval blocks are
        ascending and non-overlapping, so no coalescing sort can reorder the
        domain out from under the data (adjacent blocks, ``next.lo ==
        prev.hi + 1``, merge without disturbing the ascending word sequence).
        An empty domain contributes no intervals, so ``concat([empty, full])``
        returns ``full`` — the issue #4 "empty composes through concat"
        contract, now honored on the default index.

        Overlapping, interleaved, or reversed domains — and any request that
        carries explicit ``positions`` (xarray's interleave path) — raise a
        pointed ``NotImplementedError`` naming the ``index_kind="pandas"``
        escape, whose materialized coordinate concatenates arbitrarily.
        """
        escape = (
            'reopen the datasets with index_kind="pandas" to concatenate them '
            "(a materialized coordinate concatenates in any order)"
        )
        if positions is not None:
            raise NotImplementedError(
                "MortonMocIndex.concat cannot honor explicit positions (an "
                f"interleave would reorder the interval domain); {escape}"
            )
        first = indexes[0]
        orders = {index._ranges.cell_order for index in indexes}
        if len(orders) != 1:
            raise NotImplementedError(
                f"MortonMocIndex.concat across mixed cell orders {sorted(orders)} "
                f"is not supported; {escape}"
            )
        blocks = [index._ranges.intervals for index in indexes]
        combined = np.concatenate(blocks) if blocks else np.empty((0, 2), dtype=np.uint64)
        # The data variables concatenate in block order; the fabricated
        # coordinate has to match row for row. That holds only if the blocks
        # are ascending and disjoint end to end — otherwise MortonRanges'
        # coalescing sort would reorder the domain out from under the data. A
        # single vectorized check covers within-block (already disjoint) and
        # cross-block boundaries: next.lo must clear prev.hi.
        if combined.shape[0] > 1 and (combined[1:, 0] <= combined[:-1, 1]).any():
            raise NotImplementedError(
                "MortonMocIndex.concat supports only disjoint, ascending domains "
                "(the batch-sweep case); these overlap, interleave, or are "
                f"reversed; {escape}"
            )
        return cls(
            MortonRanges(combined, first._ranges.cell_order),
            dim=first._dim,
            name=first._name,
        )

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        """Fabricate the coordinate variable — arithmetic, zero store reads.

        Plain numpy by default. When ``chunks`` were requested at
        construction, each chunk fabricates in its own dask task (the
        ``HealpixMocIndex`` chunked shape); dask is required only then.
        """
        attrs = encoding = None
        if variables is not None and self._name in variables:
            attrs = variables[self._name].attrs
            encoding = variables[self._name].encoding
        if self._chunks is None:
            data: Any = self._ranges.fabricate()
        else:
            try:
                import dask
                import dask.array as da
            except ImportError as exc:  # pragma: no cover - dask-less env
                raise ImportError(
                    "chunked coordinate fabrication requires dask; install it or "
                    "drop the chunks= request"
                ) from exc
            chunk_arrays = [
                da.from_delayed(
                    dask.delayed(_extract_chunk)(self._ranges, slice_),
                    shape=(chunksize,),
                    dtype="uint64",
                    name=f"chunk-{index}",
                    meta=np.array((), dtype="uint64"),
                )
                for index, (chunksize, slice_) in enumerate(
                    _construct_chunk_ranges(self._chunks, self._ranges.size)
                )
            ]
            data = da.concatenate(chunk_arrays, axis=0)
        return {self._name: xr.Variable(self._dim, data, attrs=attrs, encoding=encoding)}

    def _labels_to_words(self, label) -> tuple[np.ndarray, bool]:
        """``(uint64 words, was_scalar)`` from a sel label (words/strings/lists)."""
        scalar = np.ndim(label) == 0
        values = np.asarray([label] if scalar else label).ravel()
        words = np.asarray(
            [morton_word(v.item() if hasattr(v, "item") else v) for v in values],
            dtype=np.uint64,
        )
        return words, scalar

    def sel(self, labels: dict[Any, Any], method=None, tolerance=None) -> IndexSelResult:
        """Label selection by cell id — rank arithmetic, ``KeyError`` on misses.

        Labels may be packed ``uint64`` words, decimal strings, or
        lists/arrays of either. ``method``/``tolerance`` have no meaning on
        an exact cell domain and raise; so do slices (cell ids are not an
        ordinal axis the user should be slicing by half-open label ranges —
        pass an AOI cover or use ``isel``).

        Repeated labels return repeated positions in label order — the same
        rows ``PandasIndex`` selects (the rank lookup is elementwise, so
        repeats plumb straight through the ``IndexSelResult`` positions).
        The result of a duplicated selection carries no lazy index (an
        interval set cannot hold duplicate labels) and drops it silently:
        this is the truncation-join lookup pattern
        (``coarse.sel(morton=parent_cells(fine, level))``, phase 6b), not an
        accident. A duplicate-free non-monotonic pick still warns on drop.
        """
        if method is not None or tolerance is not None:
            raise ValueError("MortonMocIndex does not support method= or tolerance=")
        label = labels[self._name]
        if isinstance(label, slice):
            raise TypeError(
                "slice selection on morton words is not supported; select with an "
                "AOI cover (open_hive(aoi=...)) or positionally via isel"
            )
        words, scalar = self._labels_to_words(label)
        positions = self._ranges.rank(words)
        if scalar:
            return IndexSelResult({self._dim: int(positions[0])})
        indexes: dict[Any, xr.Index] = {}
        try:
            indexes = {self._name: self._replace(self._ranges.subset(positions))}
        except ValueError:
            # Unrepresentable pick: values still select; the drop surfaces on
            # the positional isel xarray lowers this to (single decision
            # site — duplicated lookups drop silently, reorders warn).
            pass
        return IndexSelResult({self._dim: positions}, indexes=indexes)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | xr.Variable]
    ) -> "MortonMocIndex | None":
        """Positional subset in interval algebra; ``None`` when unrepresentable.

        A non-monotonic or duplicated indexer cannot round-trip an interval
        set, so the lazy index drops (values stay correct) — this is also
        where a ``sel`` drop surfaces, since xarray lowers label selection to
        a positional ``isel``. A duplicate-free non-monotonic pick warns on
        the drop; *any* duplicate-containing pick drops *silently* — an
        interval set cannot hold duplicates, and warning on every truncation-
        join lookup (phase 6b) would be noise, so a user-initiated duplicate
        ``sel(morton=[w, w])`` is included in that silence too (see
        :func:`_warn_index_dropped`). A scalar collapse returns ``None``
        silently — the dimension is gone, so there is no index to keep.
        """
        indexer = indexers[self._dim]
        if isinstance(indexer, xr.Variable):
            indexer = indexer.data
        if np.ndim(indexer) == 0 and not isinstance(indexer, slice):
            return None  # scalar collapse: the dimension is gone
        try:
            return self._replace(self._ranges.subset(indexer))
        except ValueError:
            # Slices never carry duplicates (subset only rejects reversed
            # steps), so the silent duplicate path is array-shaped only.
            if not isinstance(indexer, slice):
                positions = np.asarray(indexer, dtype=np.int64).ravel()
                positions = np.where(positions < 0, positions + self.size, positions)
                if np.unique(positions).size < positions.size:
                    # Any duplicate-containing pick drops silently, incl. a
                    # user-initiated sel(morton=[w, w]) — interval sets cannot
                    # hold duplicates, so there is nothing to warn toward.
                    return None
            _warn_index_dropped()  # duplicate-free non-monotonic picks
            return None

    def equals(self, other: xr.Index, **kwargs) -> bool:
        """Domain equality; pairing with a non-MOC index raises pointedly."""
        self._check_pairable(other)
        return self._dim == other._dim and self._ranges.equals(other._ranges)

    def join(self, other: "MortonMocIndex", how: str = "inner") -> "MortonMocIndex":
        """Alignment in interval algebra — no materialization."""
        self._check_pairable(other)
        if how == "inner":
            return self._replace(self._ranges.intersection(other._ranges))
        if how == "outer":
            return self._replace(self._ranges.union(other._ranges))
        if how == "left":
            return self._replace(self._ranges)
        if how == "right":
            return self._replace(other._ranges)
        raise ValueError(f"unsupported join method {how!r}")

    def reindex_like(self, other: "MortonMocIndex") -> dict[Hashable, Any]:
        """Positions of ``other``'s cells in this domain (``-1`` where absent)."""
        self._check_pairable(other)
        return {self._dim: self._ranges.rank(other._ranges.fabricate(), missing=-1)}

    def _check_pairable(self, other: object) -> None:
        if not isinstance(other, MortonMocIndex):
            raise TypeError(
                f"cannot combine a MortonMocIndex with {type(other).__name__}: the "
                f"lazy (interval) and pandas-backed morton indexes share no alignment "
                f"currency. Reopen both datasets with the same index_kind "
                f'("moc" or "pandas").'
            )

    def __repr__(self) -> str:
        return (
            f"<MortonMocIndex(level={self.level}, ranges={len(self._ranges.intervals)}, "
            f"size={self.size})>"
        )

    def _repr_inline_(self, max_width: int) -> str:
        return f"MortonMocIndex(ranges={len(self._ranges.intervals)}, size={self.size})"

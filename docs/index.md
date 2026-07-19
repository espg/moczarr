# moczarr

Sparse-DGGS xarray reader for **morton-hive** zarr stores: MOC-declared
domains, arithmetic shard paths, lazy dense views.

`moczarr` opens stores written by [zagg](https://github.com/englacial/zagg)
under the morton-hive layout convention
([espg/mortie#62](https://github.com/espg/mortie/issues/62)). See the
[plan issue](https://github.com/espg/moczarr/issues/1) for the roadmap and the
[zagg design doc](https://github.com/englacial/zagg/blob/main/docs/design/sparse_coverage.md)
for the architecture this implements (§5 reader, §6 xarray extension).

## Status

Pre-alpha: the convention core (hive paths, manifest, coverage envelopes) is
implemented; the store layer and `open_hive()` are in progress.

The **`cell_ids` fabrication layer** is in: under the morton-only storage
decision ([englacial/zagg#262](https://github.com/englacial/zagg/issues/262),
"NESTED is fabricated, never stored") zagg stops writing the NESTED
`cell_ids` array, and `open_hive()` derives it exactly from the stored
`morton` coordinate instead (`fabricate_cell_ids="auto"`: stored arrays are
kept while dual-write continues; morton-only stores get the fabricated
view). This layer is the ratified gate for zagg's writer flip. The
order-29→24 clip policy for browser/float64 safety is pending the
resolution-discriminator metadata on the zagg#262 thread.

**Shared store handle + concurrent metadata**
([issue #5](https://github.com/espg/moczarr/issues/5)): one `open_hive()`
call now constructs exactly one obstore store and one zarr wrapper — every
read (manifest, root MOC, stamps, leaf data) flows through that pair, with
leaf groups opened by deep path through the parentless digit tree. The
candidate leaves' stamp GETs and the discovery walk's per-level LISTs batch
concurrently behind `open_hive(..., concurrency=32)` (the zarr-python knob
vocabulary; `None` or `1` runs serially, and the batching is notebook-safe
under a running event loop). Leaf *data* opens remain serial pending the
lazy-index work; `tools/bench_open.py` measures both.

## The MOC-backed lazy index (`index_kind="moc"`)

`open_hive(..., index_kind="moc")` replaces the read-and-materialize
coordinate posture with the §6 lazy one: the row domain is held as a
`MortonRanges` interval set built from the SAME coverage arithmetic that
selected the leaves (root MOC ranges — or the walked shard list on
fallback — intersected with the AOI in interval space), and the `morton`
coordinate is *fabricated* from it by a `MortonMocIndex`
(`moczarr.moc_index`, a plain `xarray.Index` — no xdggs needed). The
on-disk `morton`/`cell_ids` arrays are never read: `tools/bench_open.py`
pins **zero coordinate-chunk GETs** for a moc open, and the result is
value-identical to the `"pandas"` path (data vars, fabricated morton,
fabricated NESTED `cell_ids`) across AOI, walk-fallback, and morton-only
stores.

The substrate works in **rank space**, not word space: packed words at a
fixed cell order are not unit-stride across a shard subtree, but
`word >> shift` (the base+digit field) is a global unit-stride rank
coordinate, contiguous across rank-consecutive shards — so shard subtrees,
AOI intersection, `sel`/`isel`, and alignment are all searchsorted/cumsum
arithmetic on `[lo, hi]` interval pairs. The `(shift, marker)` packing
parameters are probed from mortie at construction and span-checked, so a
packing change fails loudly.

`sel` accepts packed `uint64` words, decimal strings, or lists of either;
selections an interval set cannot represent (scalar collapse, non-monotonic
picks) degrade by dropping the lazy index — never by wrong answers. With
`decode=True` the lazy index is wrapped by the xdggs `MortonIndex`
(`index_kind="moc"`, mirroring upstream `HealpixIndex`), so the `ds.dggs`
accessor works unchanged. Mixing moc- and pandas-indexed datasets in one
alignment raises pointedly: reopen both with the same `index_kind`.

The default stays `index_kind="pandas"` for this phase; the flip to
`"moc"` is a phase-7 task (ratified on
[issue #1](https://github.com/espg/moczarr/issues/1)). Mixed-order
(pyramid) domains are out of scope v1 — intervals-per-order is the named
seam, gated with [issue #8](https://github.com/espg/moczarr/issues/8) on
mortie#116.

## The cross-resolution truncation join

Morton nesting makes cross-resolution work a *lookup*, not machinery: a
fine cell's ancestor at any coarser order is a string-prefix truncation of
its decimal id (§5's `fine_id.startswith(coarse_id)` predicate), computed
on packed words by one vectorized `clip2order` call. Two functions in
`moczarr.join` (settling O4 on
[englacial/zagg#198](https://github.com/englacial/zagg/issues/198)) expose
it — say ICESat-2 heights aggregated at order 8 and a coarser GEDI-style
product at order 6, each opened with `open_hive` from its own store:

```python
import moczarr

fine = moczarr.open_hive("s3://bucket/icesat2-hive", anonymous=True)     # order 8
coarse = moczarr.open_hive("s3://bucket/gedi-hive", anonymous=True)      # order 6

# Coarse → fine lookup: for each fine cell, what does its containing
# coarse cell say?  Coarse variables land on fine's cells dimension.
both = moczarr.join_coarse(fine, coarse, suffix="_o6")
anomaly = both["h_mean"] - both["h_mean_o6"]
```

`join_coarse` derives coarse's level from its `morton` coordinate (or lazy
index), truncates fine's cells to that level, and looks each parent up in
coarse — a values-level operation, so any pairing of `index_kind`s (or no
index at all) joins identically, and it never sees paths or leaf basenames
(two stores today, two products of one multi-product store after
englacial/zagg#299 — tracked in
[issue #11](https://github.com/espg/moczarr/issues/11)). `how="left"`
(default) keeps every fine row, filling uncovered rows with xarray's
missing-value semantics (floats get NaN; integers promote to float64);
`how="inner"` drops them and preserves integer dtypes. Name collisions
always raise with the suggested `suffix` — never a silent rename.

The aggregation direction ships no machinery at all: `parent_cells`
fabricates the parent coordinate and plain xarray `groupby` does the rest,

```python
parents = moczarr.parent_cells(fine, 6)          # DataArray "parent_o6"
coarse6 = fine.groupby(parents).mean()           # fine → coarse reduction
```

which is also how the join's golden test builds its coarse product before
joining it back (every fine row must recover its own group's aggregate).
Persisted coarse products are zagg's pyramid sweep
(englacial/zagg#300), not a moczarr concern. On a moc-indexed dataset both
functions are pure arithmetic — the parent coordinate fabricates from the
interval domain without reading a coordinate chunk, and a moc-indexed
coarse answers the parent lookup straight from rank arithmetic (repeated
labels included — the phase-6b duplicate-label `sel` semantics).

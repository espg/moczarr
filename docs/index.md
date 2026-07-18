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

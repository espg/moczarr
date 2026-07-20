# Concepts

moczarr is the read side of a convention specified elsewhere: the
morton-hive layout and morton decimal ids are owned by the
[mortie specification](https://github.com/espg/mortie/blob/main/docs/specification.md),
and the coverage tiers, commit-stamp semantics, and reader architecture by
[zagg's `sparse_coverage.md`](https://github.com/englacial/zagg/blob/main/docs/design/sparse_coverage.md)
(§4 coverage, §5 reader, §6 xarray extension). This page is the working
summary a reader of *this* package needs; those documents are normative.

## The hive tree

A morton-hive store is a digit tree of self-describing zarr v3 leaves:

```
{store_root}/
  morton_hive.json               <- static manifest; root-only exception
  coverage.moc                   <- root ranges MOC; root-only exception
  {sign+base}/{d1}/.../{d_n}/    <- one decimal digit per level
    {full_id}.zarr/              <- vanilla zarr v3 leaf
    {full_id}_{window}.zarr/     <- time-windowed leaf (morton-hive/2)
```

Ids are **morton decimal strings**: a sign, a base digit (`1..6`), then one
digit `1..4` per order. A string prefix is a spatial ancestor — which is
what makes shard paths, AOI covers, and cross-resolution joins arithmetic.
Below the root, a node holds only digit children and `*.zarr` objects (the
node invariant); the manifest and the root `coverage.moc` are the two
root-only exceptions.

A leaf is *complete* iff its root zarr attrs carry the commit stamp
(`morton_hive_commit`); an unstamped `.zarr/` prefix is debris a torn
worker left behind, and the reader skips it. Presence requires the stamp;
absence (a clean GET/LIST miss) is trustworthy on its own.

## Coverage tiers

Coverage — *where data exists* — is declared in tiers so a reader can
reject non-intersecting shards as cheaply as possible and only pay for
precision where the AOI actually lands:

- **Root ranges MOC** (`coverage.moc` at the root): inclusive
  `[first, last]` runs of same-order shard ids. One GET names every shard
  worth considering. It is a *cache* tier: absent or unusable, the reader
  degrades to the discovery walk — never to a wrong answer.
- **Leaf tier-0 box** (in the commit stamp): up to four morton ids
  bounding the leaf's coverage. Intersection with the AOI is a cheap
  conservative reject (false positives only).
- **Leaf bitmap** (`coverage.moc` sidecar inside the leaf, when
  `encoding: "bitmap"`): exact cell-order occupancy as a zstd-compressed
  bit field. `encoding: "full"` means the whole subtree is covered and no
  sidecar exists. A *present-but-corrupt* bitmap raises rather than
  degrading — silently zero-padding would fabricate false negatives,
  indistinguishable from healthy sparse coverage.

## Domain vs. occupancy

Two different questions, deliberately kept apart:

- The **domain** is the set of rows a dataset has: the shard subtrees the
  coverage arithmetic selected, intersected with the AOI. zagg leaves are
  dense within a shard, so the domain is pure arithmetic — every cell of a
  selected subtree is a row, occupied or not.
- **Occupancy** is data-plane: which of those rows hold observations
  (`count > 0`, non-fill values, the bitmap sidecar's exact answer).

The lazy index below indexes the *domain*. Occupancy-aware selection is an
ordinary data-plane filter (`ds.where(ds["count"] > 0, drop=True)`), not
index semantics — keeping the index in lockstep with the rows the leaves
actually store.

## The lazy index

`open_hive(..., index_kind="moc")` replaces read-and-materialize with the
§6 posture: the row domain is held as a `MortonRanges` interval set built
from the *same* coverage arithmetic that selected the leaves, and the
`morton` coordinate is fabricated from it on demand by a `MortonMocIndex`
(a plain `xarray.Index` — core, no xdggs needed). The on-disk
`morton`/`cell_ids` arrays are never read: `tools/bench_open.py` pins
**zero coordinate-chunk GETs** for a moc open, and the result is
value-identical to the `index_kind="pandas"` path.

The substrate works in **rank space**, not word space: packed words at a
fixed cell order are not unit-stride across a shard subtree, but
`word >> shift` (the base+digit field) is a global unit-stride rank
coordinate. Shard subtrees, AOI intersection, `sel`/`isel`, and alignment
are all searchsorted/cumsum arithmetic on inclusive `[lo, hi]` interval
pairs; the `(shift, marker)` packing parameters are probed from mortie at
construction and span-checked, so a packing change fails loudly instead of
mis-ranking.

Edges to know about:

- `sel` accepts packed `uint64` words, decimal strings, or lists of
  either. Selections an interval set cannot represent (scalar collapse,
  non-monotonic picks) degrade by **dropping the lazy index with a
  warning** — never by wrong answers.
- Mixing moc- and pandas-indexed datasets in one alignment raises
  pointedly: reopen both with the same `index_kind`.
- `xr.concat` of two moc-indexed datasets raises `NotImplementedError`
  (an interval-set index has no concat currency yet); open with
  `index_kind="pandas"` when concatenating across opens.
- Mixed-order (pyramid) domains are out of scope v1: intervals-per-order
  is the named seam
  ([issue #8](https://github.com/espg/moczarr/issues/8), gated on
  mortie#116).

## NESTED is fabricated, never stored

Under the morton-only storage decision
([englacial/zagg#262](https://github.com/englacial/zagg/issues/262)), the
packed `morton` word is the only stored cell identity; the HEALPix NESTED
`cell_ids` view is derived **exactly** from the words via mortie's
vectorized `mort2healpix` — so Python consumers keep NESTED for free
without the writer carrying a redundant array. `open_hive`'s
`fabricate_cell_ids="auto"` keeps a stored `cell_ids` untouched where one
exists (dual-written stores) and fabricates on morton-only stores;
fabrication is a Python-side convenience view, and the fabricated ids are
bit-identical to what a dual-writing store would have stored.

One seam: NESTED ids above order 24 exceed the float64-exact integer range
(`2**53`), so they are unsafe as JS Numbers in browser consumers.
Fabrication above order 24 currently warns; the 29→24 clip policy lands
with the resolution-discriminator metadata on the zagg#262 thread.

## Looking ahead

The next reader-model step is a DataTree-shaped `open_store()` — one child
node per product of a multi-product store, each node an `open_hive`
dataset ([issue #15](https://github.com/espg/moczarr/issues/15)).

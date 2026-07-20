# Quickstart

Every snippet on this page runs against the committed SERC fixture store
(`tests/data/serc_hive` — six order-6 shards around the NEON SERC site,
ATL06-shaped synthetic data, cells at order 8), from a repo checkout:

```sh
git clone https://github.com/espg/moczarr && cd moczarr
pip install '.[xdggs]'
```

An S3-hosted store works identically — pass `s3://bucket/prefix` (plus
`anonymous=True` for public buckets) wherever the local path appears.

## Open a store

```python
import moczarr

ds = moczarr.open_hive("tests/data/serc_hive")
```

One manifest GET bootstraps the layout, the root coverage MOC names the
shards, and the stamped leaves concatenate into one dataset in ascending
packed morton order:

```
<xarray.Dataset> Size: 5kB
Dimensions:     (cells: 96)
Coordinates:
    cell_ids    (cells) uint64 768B 238064 238065 238066 ... 239118 239119
    morton      (cells) uint64 768B 5340987683084697608 ... 5359547439361556488
Data variables:
    count       (cells) int32 384B 30 31 6 2 11 9 22 34 30 ... 0 0 0 0 0 0 0 0
    h_mean      (cells) float32 384B 29.92 22.49 22.39 28.04 ... nan nan nan nan
    ...
```

`morton` is the packed `uint64` morton word (the stored cell identity);
`cell_ids` is the HEALPix NESTED view, fabricated exactly from the words
when the store doesn't carry it (see
[fabrication](concepts.md#nested-is-fabricated-never-stored)). The
manifest summary rides along as `ds.attrs["morton_hive"]`.

## Scope to an AOI

An AOI is a morton cover: packed words or decimal strings, mixed orders
allowed. Because a decimal prefix is a spatial ancestor, one coarse id
covers its whole subtree — here an order-5 cell that contains three of the
fixture's six order-6 shards:

```python
sub = moczarr.open_hive("tests/data/serc_hive", aoi=["433142"])
sub.sizes  # {'cells': 48} — half the store never opened, rows exact
```

Shards outside the cover are excluded arithmetically (coverage-MOC
intersection — leaves not touched), and rows are trimmed exactly within
partially-covered shards. An AOI that intersects no coverage is a data
answer, not an error: you get a schema-correct **empty** dataset and a
`UserWarning` (only a store with no stamped coverage anywhere raises
`NoCoverageError`).

## The lazy index

`index_kind="moc"` holds the row domain as an interval set instead of
reading the stored coordinate — the on-disk `morton`/`cell_ids` chunks are
**never fetched**, and the coordinate fabricates on demand:

```python
lazy = moczarr.open_hive("tests/data/serc_hive", index_kind="moc")
lazy.xindexes["morton"]  # <MortonMocIndex(level=8, ranges=5, size=96)>
```

The result is value-identical to the materialized (`index_kind="pandas"`)
open; `sel`/`isel`/alignment run as rank arithmetic on intervals. See
[the lazy index](concepts.md#the-lazy-index) for what degrades (and how)
when a selection can't be represented as intervals — and note that
`xr.concat` of two moc-indexed datasets is not supported: open with
`index_kind="pandas"` when you need to concatenate across opens.

## Decode: the `ds.dggs` accessor

With the `[xdggs]` extra, `decode=True` registers the `"morton"` grid on
the result, enabling xdggs's accessor ops:

```python
ds = moczarr.open_hive("tests/data/serc_hive", decode=True)

near = ds.dggs.sel_latlon([37.356], [-75.937])   # point -> containing cell
int(near["count"].values[0])                     # 30

boundaries = ds.dggs.cell_boundaries()           # shapely Polygons
boundaries.values[0].geom_type                   # 'Polygon'
```

## Cross-resolution: aggregate up, join down

Morton nesting makes cross-resolution work a lookup. **Fine → coarse** is
plain xarray groupby over the fabricated parent coordinate:

```python
parents = moczarr.parent_cells(ds, 6)     # DataArray "parent_o6"
coarse = ds.groupby(parents).mean().rename({parents.name: "morton"})
coarse.sizes                              # {'parent_o6': 6} -> 6 shards
```

**Coarse → fine** is `join_coarse`: each fine cell looks up its containing
coarse cell, and the coarse variables land on the fine cells dimension:

```python
both = moczarr.join_coarse(ds, coarse, variables=["h_mean"], suffix="_o6")
anomaly = both["h_mean"] - both["h_mean_o6"]   # fine minus its shard mean
```

The coarse dataset can equally be a *second store* at a coarser order
(`moczarr.open_hive("s3://.../gedi-hive")`) — the join operates on
coordinate values, never on paths, so any pairing of stores and index
kinds joins identically. `how="left"` (default) keeps every fine row
(uncovered rows fill per xarray missing-value semantics); `how="inner"`
drops them and preserves integer dtypes. Name collisions always raise with
a suggested `suffix` — never a silent rename.

## Time-windowed stores

A `morton-hive/2` store shards leaves by window label; pass `window=` to
scope. Omitting it on a windowed store raises with the labels that exist:

```python
ds_2019 = moczarr.open_hive("s3://bucket/windowed-hive", window="2019")
```

## Where next

- [Concepts](concepts.md) — the hive tree, coverage tiers, and the lazy
  index, with pointers to the normative specs.
- [API reference](api/open.md) — `open_hive` and the full module surface.
- [Development](development.md) — regenerating the fixture, the
  golden-vector policy.

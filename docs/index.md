# moczarr

Sparse-DGGS xarray reader for **morton-hive** zarr stores: MOC-declared
domains, arithmetic shard paths, lazy dense views.

`moczarr` opens stores written by [zagg](https://github.com/englacial/zagg)
under the morton-hive layout convention: a digit tree of self-describing
zarr v3 leaves keyed by morton decimal ids, with a static manifest
(`morton_hive.json`) and hierarchical coverage MOCs (`coverage.moc`)
declaring where data exists — so a reader intersects an area of interest
*arithmetically* instead of listing objects or materializing a global grid.

What you get:

- **`open_hive()`** — one call from store root (local path or
  `s3://bucket/prefix`) to a lazy `xarray.Dataset`, with AOI and
  time-window scoping resolved through coverage metadata, not object
  listings.
- **A MOC-backed lazy index** — the row domain held as an interval set
  built from the same coverage arithmetic that selected the leaves; the
  on-disk cell arrays are never read, and the `morton` coordinate is
  fabricated on demand.
- **Exact NESTED fabrication** — HEALPix NESTED `cell_ids` derived
  exactly from the packed morton words ("NESTED is fabricated, never
  stored").
- **Cross-resolution joins** — morton truncation makes a coarse↔fine join
  a vectorized lookup (`parent_cells`, `join_coarse`), no I/O.
- **xdggs integration** (`moczarr[xdggs]`) — a registered `"morton"` grid,
  so `ds.dggs.sel_latlon`, `cell_boundaries`, `zoom_to` work on any
  `open_hive` result.

## Install

```sh
pip install moczarr            # core reader
pip install 'moczarr[xdggs]'   # + the ds.dggs accessor integration
```

Until the first PyPI release lands, install from git:

```sh
pip install 'moczarr[xdggs] @ git+https://github.com/espg/moczarr'
```

## Quickstart

Ten lines against the in-tree SERC fixture store (run from a repo
checkout; any hive store root — including `s3://` — works the same):

```python
import moczarr

ds = moczarr.open_hive("tests/data/serc_hive")        # whole store
sub = moczarr.open_hive("tests/data/serc_hive",       # AOI-scoped:
                        aoi=["433142"])               # any morton cover

parents = moczarr.parent_cells(ds, 6)                 # fine -> coarse
coarse = ds.groupby(parents).mean().rename({parents.name: "morton"})
both = moczarr.join_coarse(ds, coarse,                # coarse -> fine
                           variables=["h_mean"], suffix="_o6")
anomaly = both["h_mean"] - both["h_mean_o6"]
```

Continue with the [quickstart](quickstart.md) (open → AOI → decode →
join, every snippet runnable against the fixture), the
[concepts](concepts.md) page for how the store convention and the lazy
index work, and the [API reference](api/open.md).

## Where the convention lives

moczarr is the *read* side of a convention owned elsewhere:

- the morton-hive layout and morton decimal ids are specified in the
  [mortie specification](https://github.com/espg/mortie/blob/main/docs/specification.md);
- the coverage tiers, commit-stamp semantics, and reader architecture are
  designed in
  [zagg's `sparse_coverage.md`](https://github.com/englacial/zagg/blob/main/docs/design/sparse_coverage.md)
  (§4 coverage, §5 reader, §6 xarray extension).

Plan and progress: [espg/moczarr#1](https://github.com/espg/moczarr/issues/1).
Next up: a DataTree-shaped `open_store()` for multi-product stores
([issue #15](https://github.com/espg/moczarr/issues/15)).

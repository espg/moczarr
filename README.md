# moczarr

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/espg/moczarr/main?labpath=docs%2Fexamples%2Fquickstart.ipynb)

Sparse-DGGS xarray reader for **morton-hive** zarr stores: MOC-declared
domains, arithmetic shard paths, lazy dense views.

`moczarr` opens stores written by [zagg](https://github.com/englacial/zagg)
under the morton-hive layout convention: a digit tree of self-describing
zarr v3 leaves keyed by morton decimal ids, with a static manifest and
hierarchical coverage MOCs declaring where data exists — so a reader
intersects an area of interest *arithmetically* instead of listing objects
or materializing a global grid.

```python
import moczarr

ds = moczarr.open_hive("s3://bucket/prefix", aoi=["433142"], anonymous=True)
```

## Features

- **`open_hive()`** — one call from store root (local path or
  `s3://bucket/prefix`) to a lazy `xarray.Dataset`; AOI covers (packed
  morton words or decimal strings, mixed orders) and time-window scoping
  resolve through coverage metadata, not object listings. An AOI over no
  coverage is a data answer: a schema-correct empty dataset, not an error.
- **A MOC-backed lazy index** — the row domain held as a rank-space
  interval set built from the same coverage arithmetic that selected the
  leaves; the on-disk cell arrays are never read, and the `morton`
  coordinate is fabricated on demand (`sel`/`isel`/alignment as interval
  arithmetic).
- **Exact NESTED fabrication** — HEALPix NESTED `cell_ids` derived exactly
  from the packed morton words ("NESTED is fabricated, never stored").
- **Cross-resolution joins** — morton truncation makes coarse↔fine work a
  vectorized lookup: `parent_cells` for fine→coarse groupby aggregation,
  `join_coarse` for the coarse→fine broadcast, no I/O.
- **xdggs integration** (`moczarr[xdggs]`) — a registered `"morton"` grid,
  so `ds.dggs.sel_latlon`, `cell_boundaries`, `zoom_to` work on any
  `open_hive` result.

Docs: [espg.github.io/moczarr](https://espg.github.io/moczarr/) —
quickstart, concepts, API reference, and the
[example notebook](https://espg.github.io/moczarr/examples/quickstart/)
(runnable on binder via the badge above).

## Install

```sh
pip install moczarr            # core reader
pip install 'moczarr[xdggs]'   # + the ds.dggs accessor integration
```

Until the first PyPI release lands, install from git:

```sh
pip install 'moczarr[xdggs] @ git+https://github.com/espg/moczarr'
```

Development (uses [uv](https://docs.astral.sh/uv/)):

```sh
uv sync --extra test --extra xdggs
uv run pytest -v
```

## The convention

moczarr is the read side of a convention owned elsewhere: the morton-hive
layout and morton decimal ids are specified in the
[mortie specification](https://github.com/espg/mortie/blob/main/docs/specification.md),
and the coverage tiers, commit-stamp semantics, and reader architecture in
[zagg's `sparse_coverage.md`](https://github.com/englacial/zagg/blob/main/docs/design/sparse_coverage.md)
(§4 coverage, §5 reader, §6 xarray extension). Plan and progress:
[espg/moczarr#1](https://github.com/espg/moczarr/issues/1).

## License

MIT

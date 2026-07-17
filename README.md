# moczarr

Sparse-DGGS xarray reader for **morton-hive** zarr stores: MOC-declared
domains, arithmetic shard paths, lazy dense views.

`moczarr` opens stores written by [zagg](https://github.com/englacial/zagg)
under the morton-hive layout convention (frozen at
[espg/mortie#62](https://github.com/espg/mortie/issues/62)): a digit tree of
self-describing zarr leaves keyed by morton decimal ids, with a static
manifest (`morton_hive.json`) and hierarchical coverage MOCs (`coverage.moc`)
declaring where data exists — so readers intersect an AOI arithmetically
instead of listing or materializing a global grid.

**Status: pre-alpha.** Plan and progress:
[espg/moczarr#1](https://github.com/espg/moczarr/issues/1). Design context:
[zagg `sparse_coverage.md`](https://github.com/englacial/zagg/blob/main/docs/design/sparse_coverage.md)
(§5 reader architecture, §6 xarray extension).

## Layers

1. **Core reader** (no xdggs dependency): `moczarr.open_hive(store_root, aoi=...)`
   → lazy `xr.Dataset`. Manifest GET → coverage-MOC ∩ AOI → hive paths →
   stamped-leaf opens → concat. As easy as `xr.open_zarr()` for the debug case.
2. **xdggs integration** (`pip install moczarr[xdggs]`): registers a
   `grid_name: "morton"` DGGS via xdggs's public registry
   (`MortonInfo`/`MortonIndex`), enabling `ds.dggs.sel_latlon`,
   `cell_boundaries`, `zoom_to`, `explore`, plus a MOC-backed lazy index
   bootstrapped from `coverage.moc` without reading cell arrays.
3. **Cross-resolution joins**: morton truncation joins between stores at
   different cell orders (prefix = ancestor; arithmetic, not I/O).

## Install

```sh
pip install moczarr          # core reader (not yet published)
pip install moczarr[xdggs]   # + xdggs accessor integration
```

Development:

```sh
uv sync --extra test
uv run pytest -v
```

## License

MIT

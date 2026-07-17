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

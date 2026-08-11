# Changelog

## Unreleased

- Two-store occupancy intersection
  ([#39](https://github.com/espg/moczarr/issues/39); consumer
  englacial/zagg#426, cross-sensor GEDI/ATL03 composition): the exact cell
  occupancy two hive stores share — root-`coverage.moc` prefilter → shared
  leaves → per-leaf exact bitmap AND, with the `encoding: "full"`
  short-circuit (no sidecar GET) and 4:1 OR-coarsening to the coarser cell
  order when the stores' orders differ (e.g. ATL03 o19 vs GEDI o18). Both
  API shapes of zagg#422 open question 7 ship behind one shared golden
  test until the measured comparison picks the public surface:
  `iter_occupancy_and` (an iterator of `(leaf_id, intersected cell words)`
  per shared leaf) and `occupancy_and` (one flat compacted MOC). Leaves
  without exact occupancy (box-only envelopes, missing sidecars, stamps
  without envelopes, or an envelope whose own `cell_order` sits below the
  harmonized order) contribute their conservative cover — the result is a
  documented SUPERSET under such leaves, with a once-per-call
  `UserWarning`; debris and absent leaves contribute nothing. `degrade=`
  chooses that default (`"conservative"`) or `"skip"` (drop such leaves, so
  everything returned is exact) or `"raise"`. Conservative covers stay
  compact MOCs through the whole intersection: `occupancy_and` never
  materializes a subtree, and `iter_occupancy_and` only where it has to
  yield the cells of a region that stayed a cover on BOTH sides.

- `convention.morton_word` now parses via mortie's **public**
  `decimal_to_word` ([#38](https://github.com/espg/moczarr/issues/38);
  espg/mortie#114/#156) instead of the deprecated private
  `_decimal_to_word`, which carried no compatibility promise across mortie
  releases, and the batched `decimals_to_words` replaces the per-label
  loops in `coverage.ranges_words` and `coverage.decode_bitmap` (one
  Python→Rust crossing per envelope instead of per shard / per occupied
  cell). Output is unchanged (pinned by the existing golden vectors plus a
  new scalar/batched parity pin); no mortie floor change.

- Span-restricted (subtree) reads for the ragged/HHDC layer
  ([#29](https://github.com/espg/moczarr/issues/29); the counterpart of
  englacial/zagg#351, normative property zagg spec §1.5 "Subtree spans"):
  `moczarr.ragged.read_ragged` and `moczarr.hhdc.read_tensors` accept
  `subtree=` — a packed morton AREA word or decimal string naming an
  ancestor — and fetch only the stored objects overlapping the subtree's
  contiguous cell span: on a sharded store the index suffix plus the
  covering inner chunks, on the flat layout the covering chunk objects
  (`read_cell`'s 2-GET pattern generalized; pinned by GET-count tests). A
  well-formed word disjoint from the axis warns once per call and yields
  nothing (the warning is the only discriminator vs "in-domain, nothing
  stored"); malformed / too-deep words raise; finer-than-chunk words raise
  pointing at `read_cell` (the ratified v1 refusal). With `block_order` the
  composed floor is `max(subtree_order, axis_root_order) <= block_order <=
  chunk_order` — blocks tile the visited span. The span grammar lives in
  `convention.normalize_subtree` / `convention.subtree_cell_span` (names
  parallel with zagg's `readers/_layout` pair; the nested-placement
  identity is checked against a written anchor word, never assumed), and
  the `moczarr[zagg]` leg pins live bit-identical parity with zagg's
  subtree reader (first released in zagg 0.42).

- New public `open_leaf(store_root, shard, ...)`: the leaf-direct twin of
  `open_hive` for the per-leaf readers (`moczarr.hhdc.read_tensors`,
  `moczarr.ragged.open_ragged`, ...). It owns the three layers a caller
  otherwise hand-assembles — the leaf path from the manifest's grammar
  (`convention.leaf_path` under the manifest's `path_grouping`), the
  transport with `open_hive`'s credential/`anonymous` policy, and the
  read-only `zarr.storage.ObjectStore` wrapper — so no caller does path
  arithmetic or bare-obstore incantation. `product=` re-roots on a D19
  product subtree (and a manifest-less multi-product root names its
  products, as `open_hive` does); `window=` runs the same seam every entry
  point does, required on a `morton-hive/2` store, refused on an unwindowed
  one, and refusing the reserved all-time token `all`
  ([#30](https://github.com/espg/moczarr/issues/30)); a `shard` at the wrong
  order (a cell id where a shard id belongs) raises against the manifest's
  `shard_order`. `manifest=` threads an already-read manifest (of the
  product subtree when `product=` is given) to skip the GET in an
  iterate-many-leaves loop, and `store=` shares the root handle for that
  read ([#5](https://github.com/espg/moczarr/issues/5)); the leaf store
  itself is always a fresh leaf-rooted open. The returned store is
  deliberately bare, so `manifest=` / `read_manifest` is also how a caller
  gets the `cell_order` its field paths need.

- Ambient AWS credentials in `open_object_store` — a behavior change for
  **every** `s3://` open (`open_hive`, `open_store`, `open_leaf`,
  `list_products`, ...). obstore's native chain reads env vars then falls
  back to EC2 instance metadata, so a laptop with only `AWS_PROFILE`/SSO set
  used to get an IMDS `HostUnreachable` instead of a credential error. When
  nothing explicit is supplied, moczarr now prefers boto3's resolver via
  `obstore.auth.boto3.Boto3CredentialProvider` (which carries the session's
  region into the store config itself). The probe is skipped whenever the
  caller settled it — any credential kwarg in any obstore spelling
  (`aws_`-prefixed, `token`/`session_token`, a `config=` dict),
  `skip_signature`, `anonymous=True`, or a custom `endpoint` (MinIO/R2 want
  their own credentials and region) — is memoized per process (a fresh
  session re-mints SSO/AssumeRole credentials, which an N-leaf loop would
  otherwise pay for N times) with a 5-minute credential lease, and degrades
  with a debug log on anything the probe raises: no boto3 installed, or
  botocore's `ProfileNotFound`/`ConfigParseError` from a stale profile
  beside perfectly valid env keys. boto3 stays a non-dependency.

- Resolution (pyramid-order) nodes in `open_store` (issue #15 phase 8b,
  completing the DataTree reader model): a product whose manifest declares
  sweep-generated overviews (zagg spec §4.5 — the reader binds
  `pyramid.overview.orders` and nothing else; `[]`/absent keeps today's
  flat product node, regression-pinned) opens as `{product}/{order}`
  children named by stored cell order — the source data as the source-order
  child (the same `open_hive` Dataset) and one node per declared overview
  order with a stamped object, each opened by the new
  `moczarr.pyramid.open_overview_order` (candidates = root-MOC source
  shards coarsened to the ancestor prefix; D4 stamp admission; issue-#4
  empty posture per node; zero chunk GETs on the moc default). `role` is
  surfaced per object under `attrs["zagg_objects"]`, checked per §4.3 with
  two severities split on §4.1's "never load-bearing": an uninterpretable
  cache object (role outside the closed vocabulary, missing/unknown-revision
  `zagg_overview`, no `cell_order`) is dropped with a warning while the node
  and the product's source nodes stand, and an off-order `cell_order` —
  interpretable and positively wrong — raises. Per-node variable sets differ
  in both directions by construction, and the selection helpers
  (`source_orders`, `overview_orders`, `finest_source_at`, `node_objects`)
  range over source-order sets keyed on per-object roles, answering about
  the store rather than the query (unaffected by `aoi`/`window`, and defined
  on a flat product node). Rows are laid down in packed-word order, so a
  store spanning northern and southern base cells labels them correctly.
  `window=` takes a declared label only: the reserved all-time token `all`
  (§4.2) is refused with a pointed error, since the `all.zarr` folds have no
  counterpart on the source axis yet. Order nodes are omitted with a warning
  — never guessed — when the root MOC is unusable, when a declared order was
  never swept, and on a `path_grouping > 1` store (the grouped ancestor-path
  convention is unsettled writer-side). The discovery
  walk now skips non-decimal overview basenames (`all.zarr`/
  `{window}.zarr` at ancestor nodes), so MOC-less pyramid stores still
  open their source. Fixture: a zagg-written overview store
  (`tools/generate_overview_fixture.py` — production write path + the
  `sweep_overviews` second pass; zagg sha in the golden sidecar), two
  overview orders plus per-window and all-time folds
  ([#15](https://github.com/espg/moczarr/issues/15)).

- Generic `zagg-ragged/1` decode layer (`moczarr.ragged`): strict-gated
  element attrs (`parse_ragged_attrs` — missing/foreign/newer spec raises,
  never half-parses), `read_ragged` whole-store sweep (sharded and flat
  geometries through one code path, one GET per stored object — the sibling
  `morton` coordinate included, metadata-bound located siblings with the
  §1.1 row-alignment check), `read_cell` random access (2 ranged GETs on a
  sharded store), non-ordinal debris under `c/` skipped with a warning, and
  zero product knowledge — element-generic over a 1-D morton cells axis
  (rect-grid ragged fields carry no per-cell morton and stay out of scope).
  HHDC tensor profile (`moczarr.hhdc`): `read_tensors`
  yields `(tensor, mask, (offset, gain), morton_index)` per coverage block
  — the englacial/zagg#336 contract, bit-identical to zagg's
  `readers.tdigest_tensor` (committed goldens + live parity) — with the
  mortie spec §8 deinterleave layout, block assembly, the three fit
  policies, and the 3-state occupancy mask decoded through moczarr's own
  coverage machinery (`has_exact_occupancy` discriminates the 2-state
  degrade). Digest algebra is imported from zagg via the new
  `moczarr[zagg]` extra, never vendored; mortie floor is now `>=0.9.3`
  (`rank_to_xy`/`xy_to_rank`). Conformance fixtures: the englacial/zagg#346
  spec vectors (vendored, branch-sourced pending merge) plus a zagg-written
  SERC strata fixture serving #19/#20/#21; `hash_arrays` now tolerates
  non-zarr sidecar objects inside a leaf (the in-leaf `coverage.moc`)
  ([#19](https://github.com/espg/moczarr/issues/19)).

- `zagg-composition/1` decoding (zagg spec §3, englacial/zagg#346):
  `unpack_composition` (uint64 words to positional u8 lanes, LSB byte
  first; a non-integer or negative `words` raises rather than coercing —
  §3/§7 fix the word as `uint64`), `counts_from_composition` (`round(k*N/255)` — exact for
  `N <= 254`, bounded `±(N/510 + ½)` estimate above, the writer's
  quantization plus this reader's own rounding), `lane_presence` (`lane > 0`,
  exact at every N by the presence floor, given the `fill_value: 0` §3
  requires of a composition array), plus the attrs binding —
  `parse_composition_attrs` (strict `zagg-composition/1` gate on both the
  `spec` marker and — per §3.3, which fixes the `/1` value — the declared
  `lanes` against the §3.1 order `COMPOSITION_LANES`; extracts
  `lanes`/`of`/`threshold`) and `named_lanes` (lanes keyed by the
  attrs-declared names, never a hardcoded order). `open_hive` enforces §3's
  `fill_value: 0` MUST on any array whose attrs carry a `composition.spec`
  block — a nonzero fill makes every unwritten cell report spurious lane
  presence, so it raises rather than degrading. No read-side merge:
  the §3.4 merge law stays zagg-owned
  ([#20](https://github.com/espg/moczarr/issues/20)).

- `open_store`: a store root as one `xarray.DataTree` — empty root
  (store-level attrs), one child node per product (each exactly that
  product's `open_hive` Dataset — its laziness, no stronger promise — with
  `semantic_hash` on the node attrs), kwargs forwarded per node with
  `window=` reaching only the windowed products, a `products=[...]`
  filter, and the bare single-product store as a valid one-child tree
  (roster `[]` + `bare`/`node` attrs, agreeing with `list_products`;
  `products=` there raises). xarray floor is now `>=2026.01.0` —
  bisected, not read off a changelog: `xr.Coordinates.from_xindex`
  (2025.03.0) is what the default `index_kind="moc"` path needs to open
  at all, and `set_xindex` over an already-indexed coordinate
  (`MortonMocIndex`'s adoption path) only works from 2026.01.0.
  Resolution (pyramid-order) nodes are designed on the concepts page but
  **not implemented** — gated on englacial/zagg#201's first overview
  fixture ([#15](https://github.com/espg/moczarr/issues/15)).
- Multi-product store roots (zagg D19, mortie spec §6.5): `list_products`
  enumerates the named products of a store root (surfacing `semantic_hash`
  and `aggregation.yaml` presence); `open_hive(..., product=...)` opens a
  named product's subtree; a multi-product root opened without `product=`
  errors with the product names; a product on an unsupported-but-well-formed
  spec (`morton-hive/3` and up) is listed with `manifest: None` rather than
  making its readable siblings undiscoverable. Bare single-product stores are
  unchanged ([#11](https://github.com/espg/moczarr/issues/11)).
- D20 stats sidecars + D22 rollups, read side: `read_stats` (per-leaf
  record), `read_stats_rollup` (swept fold at any digit node),
  `stats_sidecar_key`/`stats_sidecar_path` (spec-keyed naming incl. the
  D23 `{window}.stats.json` / `all` grammar); O11 content verification —
  `hash_arrays`/`combined_hash`/`verify_arrays` recompute per-array sha256
  over decoded values against the sidecar record. Ragged (vlen-bytes) arrays
  hash as `sha256(uint64_le(len) || payload)` per cell in flat C order — the
  recipe zagg's future O11 writer must adopt; `verify_arrays` also reports
  `combined_match` and never calls a leaf verified when its recorded
  combined hash disagrees
  ([#11](https://github.com/espg/moczarr/issues/11)).

## 0.1.0

First release — the complete phase 0–7 reader from the plan issue
([#1](https://github.com/espg/moczarr/issues/1)).

**`open_hive` defaults to the lazy index**: `index_kind="moc"` is the
default posture — a whole-store or AOI open reads *no* coordinate chunks,
holding the row domain as an interval set and fabricating the `morton`
coordinate on demand. The result is value-identical to the materialized
open; pass `index_kind="pandas"` to materialize instead. One workflow
difference to know: `xr.concat` of moc-indexed opens is supported for
disjoint, ascending domains (the batch-sweep case); overlapping or
out-of-order concat raises `NotImplementedError` and should open with
`index_kind="pandas"`.

- Scaffold + the morton-hive convention core: hive paths, manifest
  parsing, node invariant, morton decimal↔word helpers
  ([#2](https://github.com/espg/moczarr/pull/2)).
- Store layer (obstore-backed) + `open_hive()`: manifest bootstrap,
  coverage-MOC ∩ AOI shard selection, stamped-leaf opens, discovery-walk
  fallback, time-windowed (`morton-hive/2`) stores; the SERC test fixture
  generated by zagg's real writer
  ([#3](https://github.com/espg/moczarr/pull/3)).
- xdggs integration (`[xdggs]` extra): `grid_name: "morton"` registered
  via `register_dggs` — `ds.dggs.sel_latlon`, `cell_boundaries`,
  `zoom_to` on `open_hive` results
  ([#6](https://github.com/espg/moczarr/pull/6)).
- Exact NESTED `cell_ids` fabrication from packed morton words
  (`fabricate_cell_ids="auto"`) — the reader-side gate for zagg's
  morton-only writer flip (englacial/zagg#262)
  ([#7](https://github.com/espg/moczarr/pull/7)).
- Shared store handle + concurrent metadata: one obstore/zarr pair per
  open, batched stamp GETs and walk LISTs (`concurrency=32`), issue
  [#5](https://github.com/espg/moczarr/issues/5)
  ([#9](https://github.com/espg/moczarr/pull/9)).
- MOC-backed lazy index (`index_kind="moc"`): the row domain held as a
  rank-space interval set (`MortonRanges`) behind a `MortonMocIndex`
  (plain `xarray.Index`, core) — zero coordinate-chunk reads, value-
  identical to the materialized open
  ([#10](https://github.com/espg/moczarr/pull/10)).
- Cross-resolution truncation join: `parent_cells` (fine→coarse groupby
  coordinate) and `join_coarse` (coarse→fine lookup), any index-kind
  pairing ([#12](https://github.com/espg/moczarr/pull/12)).
- Empty-AOI contract: an AOI/window intersecting no coverage returns a
  schema-correct empty dataset + `UserWarning`; only a store with no
  stamped coverage anywhere raises `NoCoverageError`, issue
  [#4](https://github.com/espg/moczarr/issues/4)
  ([#13](https://github.com/espg/moczarr/pull/13)).
- Documentation site (quickstart, concepts, API reference), the
  binder-runnable example notebook, and the tag-driven publish pipeline
  (TestPyPI → PyPI via trusted publishing).

# Changelog

## Unreleased

- **mortie 1.0 is now the floor** ([#59](https://github.com/espg/moczarr/issues/59)):
  mortie 1.0.0 retired its plural batch names with no aliases
  (espg/mortie#187), so a fresh install against unpinned mortie failed at
  import (`cannot import name 'decimals_to_words'`, seen on a Binder build of
  the zagg reader notebooks). `coverage` now parses label batches through the
  array form of `decimal_to_word`, `intersect._expand_to` and `dggs.zoom_to`
  refine through `generate_morton_children`, and `pyproject` requires
  `mortie>=1.0.0`. No behaviour change **at these call sites**: both
  replacements are the same kernels under the surviving name (array-in /
  array-out, same `max_cells` polarity, same dense `(n, 4**d)` block). This is
  a claim about the migrated lines, not about mortie 1.0 as a whole — 1.0's
  other breaks (word-valued scalars becoming `np.uint64`, family-wide strict
  input validation, `MortonIndexScalar` → `MortonWord`) are absorbed here
  because `convention.morton_word` `int()`s its result, `coverage.as_moc_words`
  casts every AOI to `uint64` before any mortie call, and `numpy>=2.0` was
  already the floor.
  **Known limitation:** the optional `[zagg]` extra is broken until zagg ships
  its own mortie 1.0 migration (englacial/zagg#559) — released zagg (0.52.0)
  still does `from mortie import decimals_to_words` in `grids/morton.py`,
  which cannot resolve against the `mortie>=1.0.0` this release requires, so
  `moczarr.hhdc`'s t-digest path raises `ImportError` on `moczarr[zagg]`
  installs.

- New `moczarr.hhdc.block_rank(words, block_order)`
  ([#52](https://github.com/espg/moczarr/issues/52)): the block-local nested
  rank of each packed morton word, plus each word's own order — the decode a
  **located companion** needs and the library did not offer. On the tensor
  path a cell's rank IS its position on the cells axis, so `read_tensors`
  never decodes a word; a located companion (spec §9) carries one word per
  observation and has no cells axis to index, so a reader placing those
  observations inside a block had to recover the rank itself. zagg's
  `demo/07_minimal.ipynb` hand-rolled that decode and got it wrong twice —
  it skipped `point_to_area29`, so the order-29 POINT words' level-28/29
  digits came out of the §1 point band (`(suffix - 28) // 5` is 4..7, never
  0..3) and `rank_to_rowcol` raised *rank must lie in [0, 4)*. `block_rank`
  normalizes points to their order-29 area twin first, handles mixed orders
  in one array (a real located companion carries order-29 points alongside
  coarser area fallbacks — the committed fixture mixes orders 6, 22 and 29
  in one companion), and returns the per-word order so a caller can group by
  `order - block_order` and make one vectorized `rank_to_rowcol` call per
  depth. Vectorized over the ≤29 LEVELS, never over the words: a located
  companion is millions of words per leaf, where a per-word `morton_decimal`
  loop is not an option. `block_order` finer than any word's own order
  raises rather than truncating, and so does the `0` FILL word at every
  `block_order` — a companion's `morton` coordinate is fill-padded over its
  unwritten rows, `0` is not a word (the §1 prefix nibble is base cell + 1,
  so prefix `0` is unreachable), and at `block_order == 0` a pass-through
  would report it as a legitimate order-0, rank-0 word. Pinned against the
  `morton_decimal` digit
  oracle (digit at level `L`, minus 1, is that level's rank) across every
  order 0..29, and against the fixture's cells axis — a cell word ranked at
  the shard order reproduces its cells-axis position, which is the tie back
  to the tensor path.

- New root-level `moczarr.cell_index(store, field, morton_index, row, col)`
  ([#52](https://github.com/espg/moczarr/issues/52)): the global cells-axis
  index of a chunk-local `(row, col)` — the `read_cell` key. Every sweep
  reader reports a chunk-local position while `read_cell` addresses the
  array's GLOBAL cells axis, and feeding a bare rank to `read_cell` silently
  reads the wrong cell (it is always in range, so nothing complains). Ported
  from `zagg.readers.tdigest_tensor.cell_index`, which the waveform viewer in
  zagg's `demo/06_paired.ipynb` reaches across a package boundary for; it
  sits next to `read_cell` now. Pure addressing, with the reference's read
  posture kept: the chunk start is resolved from the sibling `morton`
  coordinate, only the array's STORED spans are searched (one small slice of
  `morton` per span — never the whole axis, and no digest bytes), and a
  coarser `block_order` block id, which names no single chunk, raises.
  `morton_index` takes either currency, a packed area word or a decimal
  string, and both are validated before the store is searched — the way
  `normalize_subtree` validates `subtree=` — so an int outside the uint64
  range, an int that is not a valid packed word, and an order-29 POINT word
  each name the CALLER's mistake instead of coming back as a missing chunk.
  The one mis-parse no guard can catch, a decimal id typed as an int whose
  §1 prefix nibble happens to be legal, is why the not-found message renders
  both currencies of what it parsed. A probe-gated parity leg runs it side
  by side with the zagg function it was ported from.

- `open_ragged`, `read_commits` and `read_leaf_metas` are now on the package
  root ([#49](https://github.com/espg/moczarr/issues/49)): each was
  importable only by module path (`moczarr.ragged.open_ragged`,
  `moczarr.store.read_commits`, `moczarr.store.read_leaf_metas`) while
  `open_leaf`, `read_ragged`, `read_cell`, `read_tensors` and the
  one-at-a-time `read_commit` — the functions they are used alongside — all
  sat on the root, so a per-leaf workflow was forced to mix two import
  spellings, and a reader reaching for the batched stamp read found only the
  singular one and wrote the N-GET loop. Same objects, no wrappers. A
  surface-consistency test (`tests/test_surface.py`) pins the public-workflow
  roster to the package root so the asymmetry cannot recur.

- Root-taking `coverage_moc` / `coverage_toc`
  ([#49](https://github.com/espg/moczarr/issues/49)): both typed casts now
  overload on the first argument — a `str` STORE ROOT fetches the envelope
  and casts (`coverage_moc(root, anonymous=True)`, one metadata GET through
  `load_root_coverage`, with `store=`/`**store_kwargs` as that function
  takes them); a `dict` envelope keeps today's behavior unchanged (store
  arguments with a dict are a `TypeError`). `coverage_toc`'s root form
  collapses its two absences — no usable root sidecar at all, and a sidecar
  with no usable temporal section — into the one `None`, stated in the
  docstring; `coverage_moc` keeps its no-absence posture and raises
  `ValueError` on a root publishing no usable envelope (the envelope is a
  regenerable cache — no empty cover is fabricated in its place). An
  UNREACHABLE store raises for both, never an absence answer: a local root
  that is not a directory is moczarr's own `FileNotFoundError` from
  `open_object_store`, a store-level failure signalling as anything other
  than a 404 propagates untouched, and a 404-shaped root (missing bucket,
  mistyped prefix — indistinguishable from an absent sidecar at the
  transport) is settled by probing `morton_hive.json` on the absence path
  only, one extra GET, a root carrying none raising the house `ValueError`
  (*not a hive store root*). Absence of coverage is not absence of a store.
  A first argument that is neither a `str` nor a `dict` (a `Path` root,
  `None`) is a `TypeError` naming the type.

- New public `candidate_shards(store_root, manifest=None, aoi=None,
  window=None, **store_kwargs)`
  ([#49](https://github.com/espg/moczarr/issues/49)): `candidate_leaves`
  returning what `open_leaf` takes — shard IDS (morton decimal strings),
  ascending in packed-word order — from the same discovery seam through one
  shared implementation, so the path view and the id view cannot disagree:
  for equal arguments against an unchanged store, `candidate_shards(...)[i]`
  names the leaf `candidate_leaves(...)[i]` locates (each call is its own
  discovery, so a tear-free pair is derived from one view, not from two
  calls). Select-then-open needs no string surgery and no knowledge of the
  path grammar (the `.zarr` suffix, the `path_grouping` node depth, the
  windowed `{id}_{window}` dialect), and on a windowed store the id stays
  the BARE shard — the window label is not part of it, so the same
  `window=` goes to `open_leaf`. A separate name rather than a flag because
  the return TYPE changes (mortie's `Moc.to_order` precedent,
  espg/mortie#197). Both candidate functions now also take `**store_kwargs`
  (forwarded to `open_object_store` — `anonymous=True`, `region=...` — the
  passthrough `load_root_coverage` and `open_leaf` already had; `store=`
  stays the share-one-handle path and wins when both are given) and make
  `manifest` optional, fetched in one extra metadata GET when omitted —
  the same one-GET-if-absent posture as `open_leaf(manifest=None)` — so
  the whole selection is one call: `candidate_shards(root, aoi=q,
  anonymous=True)`. The GET is the whole extra cost: a call now resolves
  ONE object store up front and threads it through the manifest read, the
  envelope read and the discovery walk alike, so the transport count is one
  per call whichever route runs (it was 2-3, each re-running the ambient
  credential resolution). A passed `manifest` is `parse_manifest`-validated
  like a fetched one, so the two spellings cannot disagree. One behavior
  change rides along on `candidate_leaves`: the discovery walk no longer
  names an object whose stem is an order-29 POINT id, matching the
  arithmetic route, which never could (§2/§6.6 — `convention.leaf_path`
  refuses point words).

- The MOC/TOC seam ([#45](https://github.com/espg/moczarr/issues/45)):
  moczarr decodes zagg spec §10's `zagg-coverage-toc/1` root section — the
  tier-1 per-shard envelope-word map, via the new public `TEMPORAL_SPEC` /
  `temporal_shard_words` / `temporal_keep`, the exact twins of
  `COVERAGE_SPEC` / `ranges_words` / `aoi_mask` — so a windowed candidate
  query resolves from metadata alone: `candidate_leaves(..., when=)`
  prunes listed shards whose word fails `toc_overlaps` and keeps unlisted
  ones by construction (§10.2: unlisted is *unknown*, never *empty*). Two
  typed casts put the result in the geometry/time world without an adapter
  — `coverage_moc(envelope) -> mortie.Moc` and
  `coverage_toc(envelope) -> mortie.Toc | None`, whose `None` is §10's
  absence rule (no readable section, or one listing nothing, is "publishes
  no temporal coverage", never "has no data in any window"); both raise on
  corrupt content rather than degrading. mortie floor is now `>=0.9.10`
  (the `Moc` / `Toc` coverage types of espg/mortie#197 / #199 that the two
  casts return, plus the toc word kernels the §10 seam runs on — one bump
  for both, per espg ruling 2 on the issue); the previous `>=0.9.3` floor
  covered neither. AOI/window boundary normalizers (`as_moc_words`,
  `as_toc_words`) stay internal by ruling 3.

- Spec-text pins advanced `9e11e65` → `d52e3063` (issue #43's re-check,
  performed per pinned item): `moczarr.ragged`'s §1 pin (the delta is
  §1.1's §8.3 temporal-sibling surface adopted in this release, plus §1.2
  extending its spec-owned-key discipline to `weights`, `times` and the
  companion declaration blocks; §1.3–§1.6 unchanged) and
  `moczarr.composition`'s §3 pin (byte-identical across the
  delta — pure advance). The §2 delta is itemized at `moczarr.ragged`'s pin
  rather than summarized: §2.0 is new, §2.1 rescopes its exact-count MUST to
  `counts`, and §2.2 is substantially rewritten (gaining the
  heterogeneous-orders reader MUST this layer satisfies by construction, and
  the "an absent `located` block is this section verbatim, never a refusal"
  sentence the module cites as spec text). The §2.0 `weights` declaration
  remains un-gated and is documented at the pin as the known gap issue #43
  tracks: what stands open there is the gate's implementation — a scope #43
  already prescribes, down to mirroring zagg's `check_weights_match` posture
  — not its design, and this advance records only the delta review #43's pin
  bullet calls for. §6 (`zagg-ragged/2`) is not pinned by either module but
  was read through in the same pass and is itemized at `moczarr.ragged`'s
  pin: §6.1 extends the `/2` element declaration to the `{field}_times`
  sibling and §6.3 notes that `times` and `weights` ride the `/2` migration
  unchanged; §6.2's byte identity is unmoved.

- Temporal companion channel (`zagg-toc/1`, spec §8.3; englacial/zagg#410 /
  PR #463): `moczarr.ragged` now binds and decodes a digest field's
  per-centroid temporal sibling — `read_ragged(..., times=True)` — named by
  the payload's spec-owned top-level `times` attrs key, **never** by
  reconstructing a naming convention from the field name (§8.3; the
  spec-text-only fixture's sibling is deliberately named `t_words`, nothing
  like `{field}_times`). Row alignment and the sibling's `uint64` element
  declaration are enforced, and words are yielded raw (mortie-toc/1 word
  semantics are deliberately not decoded at this layer). Companion arrays'
  §8 `temporal` / §9 `located` declaration blocks are gated by the new
  `parse_companion_attrs` — strict-checked when present (unknown spec /
  shape / grammar refuse loudly), absent never a refusal (absent `located`
  is §2.2 verbatim; both pre-declaration fixture populations stay green).
  The `tests/data/spec/temporal/` fixture is vendored whole-tree
  byte-identical from zagg `d52e3063`, and the conformance suite pins the
  leaf ingest words **byte-exactly** against `expected.json`'s decimal
  goldens. The §4.6 column's folded per-centroid companions at resolutions
  4 and 5 (espg's 2026-08-17 ruling: symmetric with located at every level)
  have no such golden — the vendored `expected.json` records no per-level
  entries — so they are pinned *at rest* by the O11 hash gate against
  `all.pyramid.stats.json` and *through the decode* by dtype, row alignment,
  the reserved-`0` exclusion and the digest's conserved total weight. The
  suite also pins the manifest's `h_tdigest` composability reclassification
  (`none` → `approximate`).

- Vendored spec fixtures refreshed to englacial/zagg `main` at `d52e3063`
  ([#43](https://github.com/espg/moczarr/issues/43) sweep, post zagg #420/#463):
  `minimal/` and `kitchen_sink/` are now **whole-tree byte-identical** to the
  zagg vectors — re-pinned `semantic_hash` and canonical granule ids
  (zagg #420), the `zagg-pyramid/2` manifest declaration, and the §4.6 leaf
  column surface (`all.pyramid.zarr` + `all.pyramid.stats.json`,
  `granules.json`) vendored beside each leaf. Every array byte the conformance
  suite decodes is unchanged. Vendoring the columns did force one reader
  change: `store.walk_leaves` now applies zagg spec §4.6's **normative** name
  seam — a basename ending `.pyramid.zarr` is a leaf column and is never
  yielded as a leaf (a column is commit-stamped like a leaf, so the walk's
  `read_commit` completeness check would otherwise wave it through). This
  supersedes the `b9347561` refresh instruction below: a downstream
  re-checking against its own copy of the zagg vectors should refresh from
  `d52e3063`.

- Vendored spec fixtures re-pinned under the **authalic** latitude convention
  ([#41](https://github.com/espg/moczarr/issues/41), tracking
  englacial/zagg#441 and mortie >=0.9.8): `kitchen_sink`'s two `*_locations`
  arrays hold different order-29 morton words, so their chunk bytes, their two
  O11 array hashes and the fixture's `combined` hash all move. Test data only —
  no reader behavior changes, and every other vendored array (the digest
  payloads, `count`, `morton`, `composition`, and all of `minimal`) is
  byte-identical across the convention change. (This entry's `b9347561` refresh
  floor is superseded by the whole-tree `d52e3063` re-vendor above — refresh
  from that sha instead.)

- Two-store occupancy intersection
  ([#39](https://github.com/espg/moczarr/issues/39); consumer
  englacial/zagg#426, cross-sensor GEDI/ATL03 composition): the exact cell
  occupancy two hive stores share — root-`coverage.moc` prefilter → shared
  leaves → per-leaf exact bitmap AND, with the `encoding: "full"`
  short-circuit (no sidecar GET) and 4:1 OR-coarsening to the coarser cell
  order when the stores' orders differ (e.g. ATL03 o19 vs GEDI o18). The
  public surface is `iter_occupancy_and` (an iterator of `(leaf_id,
  intersected cell words)` per shared leaf — zagg#422 open question 7
  ruled on the measured comparison, since the zagg#426 consumer reads per
  leaf); `occupancy_and` (one flat compacted MOC) stays exported as a
  documented convenience DERIVED from it — semantically `compress_moc`
  over that stream, kept in MOC currency so it never expands a subtree —
  and is the currency for MOC algebra (`open_hive(aoi=...)`, `moc_and`,
  `aoi_mask`). The derivation runs one way only: compaction discards the
  leaf attribution per-leaf reads need. Leaves
  without exact occupancy (box-only envelopes, missing sidecars, stamps
  without envelopes, or an envelope whose own `cell_order` sits below the
  harmonized order) contribute their conservative cover — the result is a
  documented SUPERSET under such leaves, with a once-per-call
  `ConservativeCoverageWarning` (new, exported, a `UserWarning` subclass —
  so degradation can be promoted to an error or silenced by category
  without touching every other warning); debris and absent leaves
  contribute nothing. `degrade=`
  chooses that default (`"conservative"`) or `"skip"` (drop such leaves, so
  everything returned is exact) or `"raise"`. Conservative covers stay
  compact MOCs through the whole intersection: `occupancy_and` never
  materializes a subtree, and `iter_occupancy_and` only where it has to
  yield the cells of a region that stayed a cover on BOTH sides. `aoi=` is
  contractually SHARD-level on both shapes — the cells of a kept leaf are
  not clipped, so results are a superset with respect to the AOI (false
  positives possible, false negatives impossible: zagg's AOI-overhang
  convention), and `coverage.aoi_mask` is the documented exact cell-level
  cut.

- New public `candidate_leaves(store_root, manifest, aoi=None, window=None)`
  ([#39](https://github.com/espg/moczarr/issues/39)): the leaf-discovery
  seam `open_hive` and `iter_occupancy_and` already shared, promoted from
  `open._candidate_leaves` so a reader can get a store's leaf roster —
  store-relative paths, ascending in packed-word order — without opening
  anything. Its docstring is the contract: root-`coverage.moc` arithmetic
  with the discovery walk as a semantically equivalent fallback (the two
  candidate sets differ only where a commit stamp settles it — D9/D4),
  `morton-hive/2` window selection through the one `validate_window` seam,
  and the shard-level `aoi` restriction with the overhang posture above.
  `aoi` now takes the same cover form as the rest of the public API
  (packed words or decimal strings, mixed orders).

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

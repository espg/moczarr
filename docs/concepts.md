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

`open_hive`'s default index posture (`index_kind="moc"`) replaces
read-and-materialize with the §6 posture: the row domain is held as a
`MortonRanges` interval set built
from the *same* coverage arithmetic that selected the leaves, and the
`morton` coordinate is fabricated from it on demand by a `MortonMocIndex`
(a plain `xarray.Index` — core, no xdggs needed). The on-disk
`morton`/`cell_ids` arrays are never read: `tools/bench_open.py` pins
**zero coordinate-chunk GETs** for a moc open, and the result is
value-identical to the materialized `index_kind="pandas"` path (one kwarg
away when a workflow needs it).

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
- `xr.concat` of moc-indexed datasets works when their domains are
  disjoint and in ascending word order (the batch-sweep / AOI-tile case);
  overlapping, interleaved, or reversed domains raise `NotImplementedError`
  pointing to `index_kind="pandas"`, whose materialized coordinate
  concatenates arbitrarily.
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

## The DataTree view: `open_store`

A multi-product store root (zagg D19, mortie spec §6.5) is a *directory of
stores*: each product lives under its own named prefix, and every product
subtree is a complete, unmodified morton-hive store. Two products of one
store are heterogeneous-schemas-by-construction — different variables,
different orders, different windows — which is exactly the
non-alignable-groups case `xarray.DataTree` exists for. `open_store`
returns that tree:

- **The root node is empty** — no variables, no coordinates, only
  store-level attrs (`morton_hive_store`: the store root and the product
  roster). A multi-product root carries no manifest of its own by design.
- **One child node per product**, named by its product name, each holding
  exactly the lazy Dataset `open_hive` returns for that product — same
  index posture, same laziness, same issue-#4 empty-AOI contract, applied
  *per node*. The product's D19 `semantic_hash` rides on the node's attrs.
- **A bare single-product store is the valid degenerate form**: a
  one-child tree (the child named from the manifest's dataset
  `short_name`), so tooling written against the tree shape works on any
  store.

Which axes become nodes is a design ruling (zagg's O14,
[issue #15](https://github.com/espg/moczarr/issues/15)), worth knowing as
a user because the answer is mostly *no*:

- **Products → nodes.** The only axis that is nodes today.
- **The window axis is never nodes.** Time stays a per-node dimension:
  pass `window=` and it scopes the time-windowed (`morton-hive/2`)
  products, while unwindowed products keep their whole (single) form —
  one call opens a store that mixes both.
- **The spatial digit axis is never nodes.** Mirroring the hive digit
  tree as tree levels would put millions of nodes of metadata client-side
  for zero query power the MOC arithmetic doesn't already give.
- **Resolution (pyramid order) will be nodes** — designed below,
  implemented once zagg's overview sweep ships — but the seamless
  mixed-order composite is a *computed* view, never a node (next
  section).

`open_store` is named for its scope, not its return type: `open_datatree`
is xarray vocabulary for zarr-native hierarchies, which the hive tree
deliberately is not (zagg's D12 interop hierarchy is `xr.open_datatree`'s
to open, someday, as a derived cache — the MOC-first opener stays truth).

## Resolution nodes: the design (ahead of implementation)

> **Status: design only — not implemented.** Implementation is gated on
> the first real overview fixture from zagg's second-pass sweep
> ([englacial/zagg#201](https://github.com/englacial/zagg/issues/201);
> its reader-facing decisions are ratified and recorded here so the tree
> shape is stable before any code lands). The normative pyramid-block /
> manifest declaration is a spec seam owned by
> [englacial/zagg#340](https://github.com/englacial/zagg/issues/340) —
> this reader plans against that spec, not against an implementation in
> flight. Tracked on
> [issue #15](https://github.com/espg/moczarr/issues/15) (phase 8b).

When a product carries sweep-generated overviews, the resolution axis
becomes a second node level under the product:

```
/                       <- empty root (store-level attrs)
  {product}/            <- today's product node
    {order}/            <- one node per stored order, e.g. 8/, 6/, 4/
```

- **Node layout `{product}/{order}`.** Order nodes are named by the
  integer cell order they store. The product node itself stays what it is
  today; on an overview-carrying store the source data becomes the
  source-order child rather than the product node's own dataset, riding
  the multiscale-DataTree conventions.
- **`role` attrs vocabulary.** Every order node declares its provenance
  in `attrs["role"]`, a closed two-value vocabulary: `"source"` — the
  writer's native cell order, exactly one per product; `"overview"` — a
  sweep-generated coarsening of the source (regenerable, D9 cache class).
  Selection helpers (source-vs-overview, "finest at or above order k")
  key on `role`, never on node names.
- **Roster absence is legal per node** (zagg#201's option-A ruling):
  overview nodes carry only the variables the sweep's aggregation roster
  rolls up, so sibling order nodes have **heterogeneous variable sets** —
  a variable present at the source order may be absent at coarser orders,
  and readers must treat per-node schemas as independent (they already
  are across products). Absence of a variable at an order is an answer,
  not an error.
- **The computed-compose rule.** A D24-heterogeneous product (regionally
  mixed cell orders) has *no complete single-order Dataset*: the seamless
  order-k view is a **computed compose** — coarsen-where-finer /
  passthrough-where-equal over the stored orders, built on the
  truncation-join arithmetic (`parent_cells` / `join_coarse`) — and is
  **never materialized as a tree node**. Nodes hold stored data; composed
  views are functions of the tree.

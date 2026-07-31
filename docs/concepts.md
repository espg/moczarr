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
- Mixed-order (pyramid) domains are out of scope for the **lazy index**:
  an interval set is a rank space, and rank is only defined at one order, so
  intervals-per-order is the named seam. This is no longer an upstream wait —
  mortie's mixed-order kernels shipped in 0.9.1 and
  [issue #8](https://github.com/espg/moczarr/issues/8) adopted them without
  taking the seam. The *values-level* surfaces did move: `coverage.aoi_mask`
  resolves each cell at its own order, so an undecoded mixed-order dataset
  masks correctly. What stays single-order is anything that binds a level —
  the moc interval set, the `MortonIndex` domain contract (one order, and it
  is `level`), and the fabricated NESTED `cell_ids` view.

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
  index posture, same read behavior, same issue-#4 empty-AOI contract,
  applied *per node*. "Lazy" is `open_hive`'s laziness, not a stronger
  promise: on the moc default the cell arrays are never read and `morton`
  is fabricated on demand, while a node spanning several leaves
  concatenates their data variables at open (the tree layer adds no reads
  of its own). The product's D19 `semantic_hash` rides on the node's attrs.
- **A bare single-product store is the valid degenerate form**: a
  one-child tree (the child named from the manifest's dataset
  `short_name`), so tooling written against the tree shape works on any
  store. That name is *derived*, not published: the roster stays `[]`
  (exactly what `list_products` says) and the store attrs carry
  `bare: True` + `node: <derived name>` — so `products=` has nothing to
  filter on there and raises.

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
- **Resolution (pyramid order) is nodes — when the manifest declares
  them.** A product carrying sweep-generated overviews grows one child
  per stored cell order (next section); the seamless mixed-order
  composite stays a *computed* view, never a node.

`open_store` is named for its scope, not its return type: `open_datatree`
is xarray vocabulary for zarr-native hierarchies, which the hive tree
deliberately is not (zagg's D12 interop hierarchy is `xr.open_datatree`'s
to open, someday, as a derived cache — the MOC-first opener stays truth).

## Resolution nodes

> **Status: implemented** (phase 8b of
> [issue #15](https://github.com/espg/moczarr/issues/15), against the
> first real overview fixture from zagg's second-pass sweep —
> [englacial/zagg#201](https://github.com/englacial/zagg/issues/201),
> whose reader-facing rulings this section records — and the normative
> pyramid-block declaration of the zagg store specification §4
> ([englacial/zagg#340](https://github.com/englacial/zagg/issues/340)).
> The committed fixture is zagg-written end to end:
> `tools/generate_overview_fixture.py`, production write path plus the
> `sweep_overviews` second pass, zagg sha recorded in the golden sidecar.)

When a product carries sweep-generated overviews, the resolution axis
becomes a second node level under the product:

```
/                       <- empty root (store-level attrs)
  {product}/            <- today's product node
    {order}/            <- one node per stored order, e.g. 8/, 6/, 4/
```

### Node layout `{product}/{order}`

Order nodes are named by the integer cell order they store (the §4.4
constant-depth rule maps each declared ancestor order `k` to cells at
`c - (s - k)`). On an overview-carrying store the product node becomes an
empty intermediate carrying the product's identity attrs (`semantic_hash`,
the manifest summary); the source data becomes the source-order child —
the same lazy `open_hive` Dataset, plus per-object role entries (below) —
and each declared overview order with at least one stamped object becomes
a sibling node, riding the multiscale-DataTree conventions. A product
whose manifest declares no overview family keeps today's flat form,
unchanged. Windowed products inherit window naming (D23): `window=` scopes
each order node to that window's `{window}.zarr` overviews, so one call
still opens a store mixing windowed and unwindowed products.

`window=` takes a **declared window label only**. The reserved all-time
token `"all"` is refused: a windowed product's all-time folds do exist on
disk (`pyramid.overview.all_time`, spec §4.5 — `all.zarr` at each ancestor
node) but they are **not yet a reader surface**, because the source axis has
no all-time leaf to pair them with. Opening them alone would hand back one
tree whose source order reports 0 cells beside overview orders summing every
window, so `open_store(..., window="all")` raises and names the gap instead.
`all` is excluded from the window grammar forever (§4.2), so the eventual
surface will be its own opt-in rather than a window label.

`aoi=` and `window=` scope **rows**, never the tree's shape and never the
answers below: an out-of-coverage AOI empties each node schema-correct
(issue #4) while `source_orders` / `overview_orders` keep reporting what the
store holds.

### `role` is per *object*, never per node

zagg#201's
[ruling (5)](https://github.com/englacial/zagg/issues/201#issuecomment-4934715001)
refuses to let position — or any node-level summary — stand in for it:

> The manifest's pyramid section additionally enumerates which orders
> *carry* overviews (a useful walker hint), but it cannot be authoritative
> per-object: an order can host both overviews and coarse *source* in
> sparse regions (D11), so walkers at overview-carrying orders still check
> `role` on open. Two artifacts, two jobs: MOC = source domain; `role` =
> per-object classification.

So the attr lives on each **leaf**, a closed two-value vocabulary:
`"source"` — writer-native data at that order; `"overview"` — a
sweep-generated coarsening of the source (regenerable, D9 cache class). It
never rides alone: #201/D11 make the **source order** and the
**aggregation method** mandatory companions in the same attrs, and those
are what an "is this comparable to my source data?" check actually reads.

An order node is in general a *mixture* of both roles — overviews
coarsened from finer regions sitting beside sparse regions written
natively at that order ("a shallow zarr may equally be *coarse source*")
— so a node-level `role` is at best a derived summary and would need a
third `"mixed"` state; this reader will not synthesize one. Selection
helpers (source-vs-overview, "finest at or above order k") key on per-leaf
`role`, never on node names and never on a node-level attr.

Concretely: every order node's dataset carries `attrs["zagg_objects"]` —
one entry per stamped object the open admitted, with the object's own
`role` (absence on disk reads as `"source"`, per the spec) and, for
overviews, its full `zagg_overview` provenance block (the D11 companions:
source cell order, per-field `class`/`method`/`nan_policy`). The contract
is checked on open, with two severities split on §4.1's "overviews are
**regenerable caches**, never load-bearing… a reader MUST NOT require them":

- **Uninterpretable** cache object — a third `role` value, a missing or
  unknown-revision `zagg_overview` block, a block without `cell_order`: the
  object is dropped with a `UserWarning` and the node keeps its remaining
  objects. Failing the whole `open_store` would take the product's *source*
  nodes down with one stale cache object, which is exactly what §4.1
  forbids; unstamped overview prefixes are already debris and skipped the
  same way (D4), exactly as for leaves.
- **Interpretable but wrong** — a provenance block positively declaring a
  `cell_order` other than the node's: this **raises**. Those rows would be
  mis-ranked under the node's §4.4 coordinate, a wrong answer rather than a
  missing one, and an off-order fold indicts the sweep rather than one
  object.

Both severities are evaluated for every *stamped* object the order node
names, before any `aoi` scoping — so what a store reports about its own
integrity does not depend on the query that opened it.

### Source is not a single order — uniqueness is per (shard, window)

D24 makes resolution heterogeneity *regional*, so `"source"` leaves can
sit at several orders in one product tree
([zagg D24](https://github.com/englacial/zagg/blob/main/docs/design/sparse_coverage.md),
§7):

> one product tree may carry **regionally heterogeneous resolution** (e.g.
> o19 cells in polar shards, o17 mid-latitude)

> (2) Per (shard, window) there is **one resolution at a time**:
> heterogeneity is regional, across shards.

A product with o19 polar shards and o17 mid-latitude shards therefore has
`{product}/19` *and* `{product}/17` carrying `role: "source"` leaves —
"the source node" is not a well-formed request. Helpers are defined over
the *set* of source orders: `source_orders` / `overview_orders` (every
stored order carrying at least one object of that role, keyed on the
per-object entries) and `finest_source_at` ("the finest source order at or
above order k" — at-or-coarser numerically), never "the one source node".

These answer about the **store**, not about the query: they are unaffected
by the `aoi`/`window` the tree was opened with, and they are defined on a
flat (declared-off) product node too, where the product node itself holds the
data — `source_orders` is its own cell order, `overview_orders` is `()`. An
empty `overview_orders` therefore always means "this product stores no
overviews", never "the AOI missed them"; and a flat node's `()` overviews
sits beside a real source order rather than reading as "no data here".
This is the same fact the computed-compose rule below rests on (a
D24-heterogeneous product has no complete single-order Dataset precisely
because its source spans orders).

### Per-node variable sets differ in *both* directions

Sibling order nodes have **heterogeneous variable sets**, and readers must
treat per-node schemas as independent (they already are across products).
zagg#201 ratified two mechanisms, one per direction — the
[recorded ruling](https://github.com/englacial/zagg/issues/201#issuecomment-5025509889)
and the
[full option space](https://github.com/englacial/zagg/issues/201#issuecomment-5025519604):

**Absent above its native order (option A — the default).** Non-composable
(`none`-class) fields, roster-kind ragged among them, are excluded per
field: "composable fields roll up, `none` fields exist only at native
resolution." A variable present at a source order may simply not exist at
coarser orders. Absence of a variable at an order is an answer, not an
error.

**Present *only* above its native order (option B — the ratified
opt-in).** An

> **explicitly declared derived summary** is available as the opt-in: e.g.
> an auto-digest of a roster field's raw values **under a different field
> name** at overview orders, so overview schema never silently differs
> from source

so an overview node can carry `X_digest` / `X_hist` / `X_count` that exist
at **no** source order — plus, under the espg-flagged opt-in Phase F,
seeded-reservoir sample fields marked `sampled: k` with their seed rule in
attrs. `tree[product]["18"].ds` having a variable
`tree[product]["20"].ds` lacks is the expected case, not corruption.

Reader-facing corollary: those declarations live in the manifest's
**pyramid block, never the semantic core** — "two products differing only
in overview-summary declarations are the same product" (D24) — so the
`semantic_hash` on a product node says nothing about which overview
variables exist. Ask the node, not the hash.

### The computed-compose rule

A D24-heterogeneous product (regionally mixed cell orders) has *no
complete single-order Dataset*: the seamless order-k view is a **computed
compose** — coarsen-where-finer / passthrough-where-equal over the stored
orders, built on the truncation-join arithmetic (`parent_cells` /
`join_coarse`) — and is **never materialized as a tree node**. Nodes hold
stored data; composed views are functions of the tree.

### Node *discovery* is manifest declaration

The once-open 8b question resolved with the
[zagg#340](https://github.com/englacial/zagg/issues/340) store
specification (§4.5): the reader binds **`pyramid.overview.orders`** and
nothing else. `[]` — or no `pyramid` block at all (pre-pyramid manifests)
— is the declared-off form and means no other key of the block may be
assumed; when `orders` is non-empty, `spacing`/`all_time`/`fields` must
all be present (additional keys are tolerated — the sweep's `materialized`
actuals, per-field `nan_policy`, `summarize`). The other candidate
mechanisms lost on spec-first grounds: a **listing walk** is what the
whole convention exists to avoid, and a **MOC extension** (per-order
coverage MOCs, the intervals-per-order seam of
[issue #8](https://github.com/espg/moczarr/issues/8)) stays unbuilt.

Declaration names the *orders*; the *objects* at each order are then named
arithmetically. Per zagg#201's
[ruling (5)](https://github.com/englacial/zagg/issues/201#issuecomment-4934715001)
the root MOC is the **source** domain —

> `coverage.moc` is built from the dispatcher's *source-shard* completion
> list (D8), so arithmetic readers driven by manifest+MOC never visit
> overview nodes at all — zero opens spent filtering.

— which keeps arithmetic source reads overview-blind (a feature), and
gives the order-node opener its candidates for free: the source shards
coarsened to the declared ancestor prefix name every node that could hold
an overview, one stamp GET each — a *different* code path from the leaf
arithmetic phase 8a wraps, sharing its tiers (D4 stamps, the issue-#4
empty posture per node, zero chunk reads on the moc default). Two
consequences worth knowing: without a usable root MOC the order nodes are
omitted with a warning (overviews are D9 regenerable caches, never
load-bearing — the source child still opens via the discovery walk, which
skips the non-decimal overview basenames); and a declared order with no
stamped object anywhere (not yet swept, or deleted) is likewise omitted
loudly rather than fabricated empty.

A third omission, for the same reason: a store declaring `path_grouping > 1`
(D21). A grouped tree's directories exist only at multiples of the grouping,
so an ancestor order that is not a group boundary has no node in it —
`4/33` would be a *truncated* group component where the tree's component is
`331` — while zagg's sweep writes ancestor nodes one component per digit
regardless of grouping. There is no settled path to name, so the reader
names none: the order nodes are omitted with a warning and the source node
stands. Every store written today is `path_grouping: 1`, where the per-digit
form *is* the tree's own node path.

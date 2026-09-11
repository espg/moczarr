"""The per-order data model: one resolution level, one Dataset (issue #37).

Runs against real writer bytes on both grammars: the vendored spec §7
fixtures (``tests/data/spec/*`` — ``zagg-pyramid/2`` declared, §4.6 columns
present, deliberately unswept) for the column tier, and the zagg-swept
``tests/data/overview_hive`` for the ``/1`` overview tier. The model's one
law under test: whatever artifact kind materializes a resolution — source
leaves, a leaf column's groups, an overview artifact — the reader gets the
SAME Dataset shape (``cells`` dim, ``morton``/``cell_ids`` coords, dense
fields dense, ragged digests as encoded vlen variables), and the exact
fold classes agree across levels (the true-downsampling doctrine, §4.4).
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moczarr import (
    open_column_order,
    open_hive,
    open_level,
    pyramid_levels,
    read_manifest,
    read_ragged,
)
from moczarr.convention import morton_decimal, morton_word
from moczarr.moc_index import MortonMocIndex
from moczarr.pyramid import OBJECTS_ATTR
from moczarr.ragged import decode_cell, parse_ragged_attrs

SPEC = Path(__file__).parent / "data" / "spec"
TEMPORAL = str(SPEC / "temporal")
OVERVIEW = Path(__file__).parent / "data" / "overview_hive"
COLUMN_REL = "1/1/2/1/3/all.pyramid.zarr"


class TestPyramidLevels:
    def test_v2_table_is_the_full_ladder(self):
        # /2: native + declared leaf resolutions (columns) + the fixed
        # every-order ladder (overviews), keyed by CELL order, finest first.
        levels = pyramid_levels(read_manifest(TEMPORAL))
        assert list(levels) == [6, 5, 4, 3, 2, 1]
        assert levels[6] == {"cell_order": 6, "order": 4, "artifact": "source"}
        assert levels[5] == {"cell_order": 5, "order": 4, "artifact": "column"}
        assert levels[4] == {"cell_order": 4, "order": 3, "artifact": "overview"}
        assert levels[1] == {"cell_order": 1, "order": 0, "artifact": "overview"}

    def test_v1_table_is_constant_depth(self):
        # /1 (the live-store grammar, first-class): declared ancestor orders
        # at the §4.4 constant-depth cell orders; no column tier exists.
        levels = pyramid_levels(read_manifest(str(OVERVIEW / "atl06")))
        assert list(levels) == [8, 6, 4]
        assert levels[8]["artifact"] == "source"
        assert levels[6] == {"cell_order": 6, "order": 4, "artifact": "overview"}
        assert levels[4] == {"cell_order": 4, "order": 2, "artifact": "overview"}
        assert not any(rec["artifact"] == "column" for rec in levels.values())

    def test_declared_off_is_the_one_level_table(self):
        # A declared-off pyramid (the legacy placeholder included) still has
        # its native level: every store is at least single-resolution.
        levels = pyramid_levels(read_manifest(str(Path(__file__).parent / "data" / "serc_hive")))
        assert levels == {8: {"cell_order": 8, "order": 6, "artifact": "source"}}

    def test_node_order_member_is_not_the_level_owner(self):
        # Resolution 4 is carried TWICE on this store: the declared node-3
        # ladder rung (the level) and the §4.6 node-order member recorded in
        # every column (the leaves' whole-footprint partials — the rung's
        # merge-source tier, never a manifest member). The table names the
        # DECLARED artifact; the partial tier stays reachable explicitly
        # through open_column_order(r == shard_order).
        levels = pyramid_levels(read_manifest(TEMPORAL))
        assert levels[4] == {"cell_order": 4, "order": 3, "artifact": "overview"}

    def test_duplicated_resolution_raises(self):
        # "One resolution names one level" is the model's addressing law: a
        # declaration placing one cell order at two nodes has no single
        # answer for open-by-resolution and is refused loudly.
        manifest = read_manifest(TEMPORAL)
        manifest["pyramid"]["overviews"][1]["cells"] = [5]  # node 3 collides with leaf 5
        with pytest.raises(ValueError, match="declared at two levels"):
            pyramid_levels(manifest)


SIBLING = "-11213"  # a SOUTHERN order-4 shard: word-after, decimal-string-before


def _write_sibling_column(root: Path, node: str = SIBLING, *, doctor=None) -> None:
    """A second leaf column at a sibling shard, by hand (spec grammar, not writer bytes).

    The vendored §7 fixtures cover exactly ONE leaf (``coverage.moc`` is the
    single range ``["11213", "11213"]``), which leaves
    :func:`open_column_order`'s headline job — assembling one resolution
    ACROSS leaves — unexercised: the multi-way concat, the ``domain.union``,
    the cross-leaf word order, and any roster longer than one. zagg has no
    committed multi-leaf ``/2`` fixture generator (the same gap the stage
    artifact records above), so the sibling is the REAL column's bytes
    re-headed at another node: its groups and ragged payloads are the
    fixture's, its ``morton`` words and ``zagg_column.node`` are the
    sibling's, and its ``count`` is scaled by ten so the two leaves are
    distinguishable by value while each leaf's own group-5 → group-4 fold
    parity stays exact (scaling is linear). Hand-built bytes are spec
    GRAMMAR, never writer bytes — the same caveat and the same deferral as
    ``_write_stage_artifact``.

    ``node`` defaults to a southern shard on purpose: its packed word sorts
    AFTER ``11213`` while its decimal string (leading ``"-"``) sorts before,
    so a decimal-ordered assembly would lay the rows down in an order the
    §4.4 moc coordinate disagrees with (the invariant ``overview_nodes``
    documents). The root MOC's new range is prepended, unsorted, for the
    same reason. ``doctor`` may mutate the sibling's root attrs before the
    stamp is re-written.
    """
    import zarr

    from moczarr.convention import column_path

    src, dst = root / COLUMN_REL, root / column_path(node)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    for r, words in ((5, [morton_word(f"{node}{d}") for d in "1234"]), (4, [morton_word(node)])):
        morton = zarr.open_array(str(dst), path=f"{r}/morton", mode="r+", zarr_format=3)
        morton[:] = np.asarray(words, dtype=np.uint64)
        count = zarr.open_array(str(dst), path=f"{r}/count", mode="r+", zarr_format=3)
        count[:] = count[:] * 10
    meta_path = dst / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta["attributes"]["zagg_column"]["node"] = node
    if doctor is not None:
        doctor(meta["attributes"])
    meta_path.write_text(json.dumps(meta))
    moc_path = root / "coverage.moc"
    moc = json.loads(moc_path.read_text())
    moc["ranges"] = [[node, node], *moc["ranges"]]
    moc_path.write_text(json.dumps(moc))


@pytest.fixture()
def two_leaves(tmp_path):
    """The temporal fixture plus a sibling leaf column — a two-leaf level."""
    root = tmp_path / "hive"
    shutil.copytree(TEMPORAL, root)
    _write_sibling_column(root)
    return str(root)


class TestMultiLeafColumnLevel:
    """The assembly tail across leaves: concat, domain union, word order."""

    def test_leaves_concat_in_ascending_word_order(self, two_leaves):
        manifest = read_manifest(two_leaves)
        ds = open_column_order(two_leaves, manifest, 5)
        assert dict(ds.sizes) == {"cells": 8}
        words = np.asarray(ds["morton"].values, dtype=np.uint64)
        # Monotonic ACROSS the leaf boundary — the law the §4.4 coordinate
        # rests on. Decimal-string order would have put the southern leaf
        # first (a leading "-" sorts before every digit).
        assert (np.diff(words) > 0).all()
        decimals = [morton_decimal(int(w)) for w in words]
        assert decimals[:4] == ["112131", "112132", "112133", "112134"]
        assert decimals[4:] == [f"{SIBLING}{d}" for d in "1234"]

    def test_domain_union_spans_both_leaves(self, two_leaves):
        manifest = read_manifest(two_leaves)
        ds = open_column_order(two_leaves, manifest, 5)
        index = ds.xindexes["morton"]
        assert isinstance(index, MortonMocIndex)
        # Two disjoint shard subtrees: the union is two intervals, 8 cells.
        assert index.ranges.size == 8
        np.testing.assert_array_equal(
            np.asarray(index.ranges.fabricate(), dtype=np.uint64),
            np.asarray(ds["morton"].values, dtype=np.uint64),
        )

    def test_roster_carries_every_admitted_leaf(self, two_leaves):
        manifest = read_manifest(two_leaves)
        ds = open_column_order(two_leaves, manifest, 5)
        entries = ds.attrs[OBJECTS_ATTR]
        assert [e["node"] for e in entries] == ["11213", SIBLING]
        assert {e["role"] for e in entries} == {"column"}

    def test_fold_parity_holds_leaf_by_leaf_and_in_total(self, two_leaves):
        # Each leaf's group 5 folds to its own group 4, and the level totals
        # agree — the §4.6 from-leaves parity, now across more than one leaf.
        manifest = read_manifest(two_leaves)
        c5 = open_column_order(two_leaves, manifest, 5)
        c4 = open_column_order(two_leaves, manifest, 4)
        assert dict(c4.sizes) == {"cells": 2}
        assert int(c5["count"].values.sum()) == int(c4["count"].values.sum())
        native_total = int(open_hive(TEMPORAL)["count"].values.sum())
        # leaf 11213 verbatim + the sibling's ten-fold copy, per leaf.
        assert [int(v) for v in c4["count"].values] == [native_total, native_total * 10]

    def test_under_coverage_is_partial_not_absent(self, two_leaves):
        # The mixed case the under-coverage posture is written for: one leaf
        # carries group 5, the other does not. The level assembles from the
        # leaf that has it and simply omits the other's footprint — absent
        # spans of the morton domain, never an error and never empty.
        path = Path(two_leaves) / COLUMN_REL / "zarr.json"
        meta = json.loads(path.read_text())
        del meta["attributes"]["zagg_column"]["groups"]["5"]
        path.write_text(json.dumps(meta))
        manifest = read_manifest(two_leaves)
        ds = open_column_order(two_leaves, manifest, 5)
        assert dict(ds.sizes) == {"cells": 4}
        assert [e["node"] for e in ds.attrs[OBJECTS_ATTR]] == [SIBLING]
        assert [morton_decimal(int(w)) for w in ds["morton"].values] == [
            f"{SIBLING}{d}" for d in "1234"
        ]

    def test_aoi_selecting_one_leaf_cuts_the_other_whole(self, two_leaves):
        manifest = read_manifest(two_leaves)
        aoi = np.array([morton_word(SIBLING)], dtype=np.uint64)
        ds = open_column_order(two_leaves, manifest, 5, aoi=aoi)
        assert dict(ds.sizes) == {"cells": 4}
        assert [morton_decimal(int(w)) for w in ds["morton"].values] == [
            f"{SIBLING}{d}" for d in "1234"
        ]


class TestOpenColumnOrder:
    def test_declared_resolution_assembles(self):
        manifest = read_manifest(TEMPORAL)
        ds = open_column_order(TEMPORAL, manifest, 5)
        # 4^(r - node) cells per column: one covered leaf at node 11213.
        assert dict(ds.sizes) == {"cells": 4}
        words = [morton_decimal(int(w)) for w in ds["morton"].values]
        assert words == ["112131", "112132", "112133", "112134"]
        assert isinstance(ds.xindexes["morton"], MortonMocIndex)
        assert "cell_ids" in ds.coords

    def test_exact_fold_parity_across_levels(self):
        # count is exact-class: the column groups and the native level agree
        # on the total — §4.6's from-leaves parity, read through the model.
        manifest = read_manifest(TEMPORAL)
        native = open_hive(TEMPORAL)
        c5 = open_column_order(TEMPORAL, manifest, 5)
        c4 = open_column_order(TEMPORAL, manifest, 4)
        total = int(native["count"].values.sum())
        assert int(c5["count"].values.sum()) == total
        assert int(c4["count"].values.sum()) == total
        # The node-order member is the leaf's whole-footprint aggregate.
        assert dict(c4.sizes) == {"cells": 1}
        assert int(c4["morton"].values[0]) == morton_word("11213")

    def test_shape_consistent_with_native_level(self):
        # The model's one law: same dims name, same coordinate set, same
        # attrs keys — a consumer holds one shape whatever the level.
        manifest = read_manifest(TEMPORAL)
        native = open_hive(TEMPORAL)
        level = open_column_order(TEMPORAL, manifest, 5)
        assert list(native.sizes) == list(level.sizes) == ["cells"]
        assert set(level.coords) == set(native.coords) == {"morton", "cell_ids"}
        # A level's variable set may be a SUBSET of the leaf's (§4.4): the
        # temporal leaf carries a none-class 'observed' that exists only at
        # native resolution; every level variable exists at the leaf.
        assert set(level.data_vars) <= set(native.data_vars)
        assert "observed" in native.data_vars and "observed" not in level.data_vars
        assert set(native.attrs["morton_hive"]) == set(level.attrs["morton_hive"])
        assert level.attrs["morton_hive"]["cell_order"] == 5
        assert level.attrs["morton_hive"]["shard_order"] == 4

    def test_ragged_rides_encoded_with_attrs(self):
        # Ragged digests are encoded vlen variables on the cells axis, the
        # §1.2 ragged block verbatim — decoded values match the
        # store-addressed reader cell for cell, companions included.
        manifest = read_manifest(TEMPORAL)
        ds = open_column_order(TEMPORAL, manifest, 5)
        assert ds["h_tdigest"].dtype == object
        assert {"h_tdigest_locations", "h_tdigest_times"} <= set(ds.data_vars)
        element = parse_ragged_attrs(ds["h_tdigest"].attrs, field="h_tdigest")
        from moczarr import open_column

        col = open_column(TEMPORAL, "11213")
        streamed = {int(w): v for w, v in read_ragged(col, "5/h_tdigest")}
        pulled = {
            int(w): decode_cell(raw, element)
            for w, raw in zip(ds["morton"].values, ds["h_tdigest"].values)
            if len(raw)
        }
        assert set(pulled) == set(streamed)
        for word, values in pulled.items():
            np.testing.assert_array_equal(values, streamed[word])

    def test_aoi_cuts_rows_exactly(self):
        manifest = read_manifest(TEMPORAL)
        aoi = np.array([morton_word("112131")], dtype=np.uint64)
        ds = open_column_order(TEMPORAL, manifest, 5, aoi=aoi)
        assert [morton_decimal(int(w)) for w in ds["morton"].values] == ["112131"]

    def test_aoi_missing_everything_is_schema_correct_empty(self):
        manifest = read_manifest(TEMPORAL)
        full = open_column_order(TEMPORAL, manifest, 5)
        aoi = np.array([morton_word("21213")], dtype=np.uint64)
        with pytest.warns(UserWarning, match="intersects no column coverage"):
            empty = open_column_order(TEMPORAL, manifest, 5, aoi=aoi)
        assert dict(empty.sizes) == {"cells": 0}
        assert set(empty.data_vars) == set(full.data_vars)
        assert {name: v.dtype for name, v in empty.data_vars.items()} == {
            name: v.dtype for name, v in full.data_vars.items()
        }

    def test_pandas_index_kind_materializes(self):
        manifest = read_manifest(TEMPORAL)
        ds = open_column_order(TEMPORAL, manifest, 5, index_kind="pandas")
        assert not isinstance(ds.xindexes.get("morton"), MortonMocIndex)
        assert ds["morton"].values.dtype == np.uint64

    def test_objects_roster_carries_the_block_verbatim(self):
        manifest = read_manifest(TEMPORAL)
        ds = open_column_order(TEMPORAL, manifest, 5)
        (entry,) = ds.attrs[OBJECTS_ATTR]
        assert entry["node"] == "11213" and entry["role"] == "column"
        block = entry["zagg_column"]
        assert block["spec"] == "zagg-column/1"
        assert set(block["groups"]) == {"5", "4"}
        assert block["groups"]["5"]["regime"] == "leaf-column"
        assert block["groups"]["5"]["merges_from_raw"] == 1

    def test_out_of_range_orders_are_refused(self):
        manifest = read_manifest(TEMPORAL)
        with pytest.raises(ValueError, match="native order is open_hive's"):
            open_column_order(TEMPORAL, manifest, 6)
        with pytest.raises(ValueError, match="ancestor artifacts"):
            open_column_order(TEMPORAL, manifest, 3)

    def test_window_seam(self):
        manifest = read_manifest(TEMPORAL)
        with pytest.raises(ValueError, match="unwindowed stores"):
            open_column_order(TEMPORAL, manifest, 5, window="2019")
        with pytest.raises(ValueError, match="reserved all-time token"):
            open_column_order(TEMPORAL, manifest, 5, window="all")
        windowed = str(OVERVIEW / "atl06_windows")
        wmanifest = read_manifest(windowed)
        with pytest.raises(ValueError, match="pass window="):
            open_column_order(windowed, wmanifest, 7)
        # A windowed store with no columns: omission with a warning, never
        # an error (§4.6 — readers never require a column).
        with pytest.warns(UserWarning, match="no stamped column"):
            assert open_column_order(windowed, wmanifest, 7, window="2019") is None

    def test_no_root_moc_degrades_to_none(self):
        # Candidates are named arithmetically (the open_overview_order
        # posture — never a listing walk on the open path); the minimal
        # fixture carries no coverage.moc.
        root = str(SPEC / "minimal")
        manifest = read_manifest(root)
        with pytest.warns(UserWarning, match="no usable root coverage.moc"):
            assert open_column_order(root, manifest, 5) is None

    def test_group_absent_from_column_is_under_coverage(self, tmp_path):
        # A stamped column whose groups map lacks the resolution contributes
        # nothing, silently (the writing declaration carried no group there
        # — §4.6 heterogeneity); with no other column the level is omitted.
        root = tmp_path / "hive"
        shutil.copytree(TEMPORAL, root)
        path = root / COLUMN_REL / "zarr.json"
        meta = json.loads(path.read_text())
        del meta["attributes"]["zagg_column"]["groups"]["5"]
        path.write_text(json.dumps(meta))
        manifest = read_manifest(str(root))
        with pytest.warns(UserWarning, match="carries resolution group 5"):
            assert open_column_order(str(root), manifest, 5) is None
        # The other group is untouched and still opens.
        assert open_column_order(str(root), manifest, 4) is not None

    def test_uninterpretable_column_is_skipped_with_a_warning(self, tmp_path):
        # The §4.1/§4.6 severity split, column side: an artifact this reader
        # cannot interpret degrades by omission (contrast the point-query
        # read_column_record, which raises).
        root = tmp_path / "hive"
        shutil.copytree(TEMPORAL, root)
        path = root / COLUMN_REL / "zarr.json"
        meta = json.loads(path.read_text())
        meta["attributes"]["zagg_column"]["spec"] = "zagg-column/9"
        path.write_text(json.dumps(meta))
        manifest = read_manifest(str(root))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert open_column_order(str(root), manifest, 5) is None
        messages = [str(w.message) for w in caught]
        assert any("zagg-column/9" in m for m in messages)
        assert any("carries resolution group 5" in m for m in messages)

    def test_wrong_identity_raises(self, tmp_path):
        # Interpretable but wrong: a column declaring another leaf's node
        # would hand back that node's cells under this shard's identity.
        root = tmp_path / "hive"
        shutil.copytree(TEMPORAL, root)
        path = root / COLUMN_REL / "zarr.json"
        meta = json.loads(path.read_text())
        meta["attributes"]["zagg_column"]["node"] = "11214"
        path.write_text(json.dumps(meta))
        manifest = read_manifest(str(root))
        with pytest.raises(ValueError, match="wrong node"):
            open_column_order(str(root), manifest, 5)

    def test_variables_are_lexically_ordered(self):
        manifest = read_manifest(TEMPORAL)
        ds = open_column_order(TEMPORAL, manifest, 5)
        assert list(ds.data_vars) == sorted(ds.data_vars)

    def test_empty_and_full_concat_compose(self):
        # The issue-#4 contract composes through concat exactly as it does
        # on the source axis: the empty side contributes no intervals.
        manifest = read_manifest(TEMPORAL)
        full = open_column_order(TEMPORAL, manifest, 5)
        with pytest.warns(UserWarning, match="intersects no column coverage"):
            empty = open_column_order(
                TEMPORAL, manifest, 5, aoi=np.array([morton_word("21213")], dtype=np.uint64)
            )
        both = xr.concat([empty, full], dim="cells")
        assert dict(both.sizes) == dict(full.sizes)
        assert {name: v.dtype for name, v in both.data_vars.items()} == {
            name: v.dtype for name, v in full.data_vars.items()
        }


class TestOpenLevel:
    """The resolution-addressed dispatcher: one cell order, one Dataset."""

    ATL06 = str(OVERVIEW / "atl06")

    def test_native_level_is_open_hive_plus_roster(self):
        from moczarr import LEVEL_ATTR

        ds = open_level(self.ATL06, 8)
        native = open_hive(self.ATL06)
        assert dict(ds.sizes) == dict(native.sizes)
        assert set(ds.data_vars) == set(native.data_vars)
        assert ds.attrs[LEVEL_ATTR]["artifact"] == "source"
        assert ds.attrs[LEVEL_ATTR]["cell_order"] == 8
        # The per-object roster rides the native level too (role absence
        # means source, surfaced per object — D11).
        assert ds.attrs[OBJECTS_ATTR]
        assert all(entry["role"] == "source" for entry in ds.attrs[OBJECTS_ATTR])

    def test_v1_overview_level(self):
        from moczarr import LEVEL_ATTR

        ds = open_level(self.ATL06, 6)
        assert ds.attrs["morton_hive"]["cell_order"] == 6
        rec = ds.attrs[LEVEL_ATTR]
        assert rec == {
            "cell_order": 6,
            "order": 4,
            "artifact": "overview",
            "spec": "zagg-pyramid/1",
            "fields": rec["fields"],
            "fold_source": rec["fold_source"],
        }
        # The declared all-fields map is the zero-open §4.4 answer: the
        # none-class entries record the absence the variable set shows.
        assert rec["fields"]["h_mean"]["class"] == "none"
        assert "h_mean" not in ds.data_vars
        assert {e["role"] for e in ds.attrs[OBJECTS_ATTR]} == {"overview"}

    def test_v2_column_level(self):
        from moczarr import LEVEL_ATTR

        ds = open_level(TEMPORAL, 5)
        rec = ds.attrs[LEVEL_ATTR]
        assert rec["artifact"] == "column"
        assert rec["spec"] == "zagg-pyramid/2"
        assert rec["fold_source"] == "cascade"
        assert dict(ds.sizes) == {"cells": 4}

    def test_declared_off_store_has_its_native_level(self):
        from moczarr import LEVEL_ATTR

        root = str(Path(__file__).parent / "data" / "serc_hive")
        ds = open_level(root, 8)
        rec = ds.attrs[LEVEL_ATTR]
        assert rec["artifact"] == "source"
        assert rec["spec"] is None and rec["fields"] is None and rec["fold_source"] is None

    def test_unknown_level_raises_with_the_table(self):
        with pytest.raises(ValueError, match=r"levels are \[8, 6, 4\]"):
            open_level(self.ATL06, 7)

    def test_shard_order_names_the_partial_tier_hint(self):
        # cell_order == shard_order on a /2 store is the §4.6 node-order
        # partial tier, not a level, whenever no ladder rung lands there —
        # the error points at open_column_order. (On the fixtures d == 1
        # puts a rung AT the shard order, so the case needs a declaration
        # with a partial ladder: legal — the recorded list is verbatim.)
        manifest = read_manifest(TEMPORAL)
        manifest["pyramid"]["overviews"] = [{"node": 4, "cells": [5]}]
        with pytest.raises(ValueError, match="open_column_order"):
            open_level(TEMPORAL, 4, manifest=manifest)

    def test_multi_product_root(self):
        root = str(OVERVIEW)
        ds = open_level(root, 6, product="atl06")
        assert ds.attrs["morton_hive"]["cell_order"] == 6
        with pytest.raises(ValueError, match="multi-product store root"):
            open_level(root, 6)

    def test_windowed_product_scopes_by_window(self):
        root = str(OVERVIEW / "atl06_windows")
        ds = open_level(root, 6, window="2019")
        assert ds.attrs["morton_hive"]["cell_order"] == 6
        with pytest.raises(ValueError, match="pass window="):
            open_level(root, 6)

    def test_every_arm_honors_the_threaded_store_handle(self):
        # issue #5, uniformly: one handle serves every level of a product,
        # the native one included. A bogus store_root proves no arm falls
        # back to constructing its own store from it (before the fold the
        # source arm did, and FileNotFoundError'd here while the column arm
        # returned rows).
        from moczarr import open_object_store

        handle = open_object_store(TEMPORAL)
        native = open_level("/nonexistent/root", 6, store=handle)
        column = open_level("/nonexistent/root", 5, store=handle)
        assert dict(native.sizes) == dict(open_hive(TEMPORAL).sizes)
        assert dict(column.sizes) == {"cells": 4}

    def test_every_arm_honors_the_threaded_manifest(self):
        # ... and the manifest passed is the manifest used, on every arm:
        # the source level used to dispatch from the given manifest while
        # reading the on-disk one, so the two could disagree about identity.
        manifest = read_manifest(TEMPORAL)
        manifest["dataset"] = {"short_name": "THREADED"}
        for r in (6, 5):
            ds = open_level(TEMPORAL, r, manifest=manifest)
            assert ds.attrs["morton_hive"]["dataset"] == {"short_name": "THREADED"}

    def test_aoi_passes_through(self):
        full = open_level(self.ATL06, 6)
        word = np.uint64(full["morton"].values[0])
        cut = open_level(self.ATL06, 6, aoi=np.array([word], dtype=np.uint64))
        assert dict(cut.sizes) == {"cells": 1}
        assert int(cut["morton"].values[0]) == int(word)


def _write_stage_artifact(root: Path, *, demotions: list | None = None) -> None:
    """A /2 ladder-rung artifact at node 1121 (order 3, cells at 4), by hand.

    zagg's staged sweep (issue #384 zagg-side) has no committed fixture
    generator yet — the same gap englacial/zagg#556 records for its /2
    validation arm — so the MODEL is pinned against spec-grammar bytes
    (§4.4: same structure as a leaf group; §4.3/§4.4: ``zagg-overview/2``
    attrs; D4 stamp last) built from the REAL column's node-order member:
    the rung at cells 4 is by law a merge of the leaves' order-4 partials,
    and with one covered leaf that merge IS the column's group-4 arrays,
    placed at the leaf's nested rank in the node's 4-cell footprint.
    Writer-bytes parity for stage artifacts stays deferred (PR checklist).
    """
    import zarr

    from moczarr import open_column

    col = open_column(TEMPORAL, "11213")
    node = zarr.open_group(str(root / "1" / "1" / "2" / "1" / "all.zarr"), mode="w", zarr_format=3)
    rank = 2  # 11213 among 1121's nested children (11211, 11212, 11213, 11214)
    words = np.array([morton_word(f"1121{d}") for d in "1234"], dtype=np.uint64)
    fields: dict[str, dict] = {}
    for name in ("count", "morton", "h_tdigest", "h_tdigest_locations", "h_tdigest_times"):
        src = zarr.open_array(col, path=f"4/{name}", mode="r", zarr_format=3)
        cell = src[:][0]  # the partial's one cell (a slice, so vlen yields bytes)
        if name == "morton":
            data, dtype, fill = words, "uint64", 0
        elif src.dtype == object:
            data = np.empty(4, dtype=object)
            data[:] = b""
            data[rank] = cell
            dtype, fill = "bytes", b""
        else:
            data = np.zeros(4, dtype=src.dtype)
            data[rank] = cell
            dtype, fill = src.dtype, src.fill_value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # zarr's vlen-bytes v3 stability note
            arr = node.create_array(
                f"4/{name}",
                shape=(4,),
                dtype=dtype,
                chunks=(4,),
                dimension_names=["cells"],
                fill_value=fill,
            )
        arr[:] = data
        arr.attrs.update(dict(src.attrs))
    record = json.loads((Path(TEMPORAL) / COLUMN_REL / "zarr.json").read_text())
    fields = record["attributes"]["zagg_column"]["fields"]
    block = {
        "spec": "zagg-overview/2",
        "node": "1121",
        "order": 3,
        "cell_order": 4,
        "source_shard_order": 4,
        "source_cell_order": 6,
        "window": "all",
        "fields": fields,
        "regime": "stage-merge",
        "merges_from_raw": 2,
        "source_children": {"folded": 1, "missing": 0, "unreadable": 0},
        "run_id": "stage-20260910T000000Z-test",
        "generation": {
            "n_leaves": 1,
            "max_leaf_timestamp": "2026-08-16T23:20:03+00:00",
            "run_ids": [],
        },
    }
    if demotions is not None:
        block["demotions"] = demotions
    node.attrs["role"] = "overview"
    node.attrs["zagg_overview"] = block
    node.attrs["morton_hive_commit"] = {
        "spec": "morton-hive/1",
        "complete": True,
        "cells_with_data": 1,
        "granule_count": 1,
        "written_at": "2026-09-10T00:00:00+00:00",
        "run_id": "stage-20260910T000000Z-test",
    }


class TestV2AncestorArm:
    """The /2 ladder-rung arm: open_overview_order(cell_order=) via open_level."""

    @pytest.fixture()
    def swept(self, tmp_path):
        root = tmp_path / "hive"
        shutil.copytree(TEMPORAL, root)
        _write_stage_artifact(root)
        return str(root)

    def test_rung_opens_by_resolution(self, swept):
        from moczarr import LEVEL_ATTR

        ds = open_level(swept, 4)
        assert dict(ds.sizes) == {"cells": 4}
        assert [morton_decimal(int(w)) for w in ds["morton"].values] == [
            "11211",
            "11212",
            "11213",
            "11214",
        ]
        rec = ds.attrs[LEVEL_ATTR]
        assert rec["artifact"] == "overview" and rec["order"] == 3
        assert rec["spec"] == "zagg-pyramid/2"
        assert ds.attrs["morton_hive"] == {
            "spec": "morton-hive/1",
            "cell_order": 4,
            "shard_order": 3,
            "dataset": read_manifest(swept).get("dataset"),
        }

    def test_v2_provenance_rides_verbatim(self, swept):
        ds = open_level(swept, 4)
        (entry,) = ds.attrs[OBJECTS_ATTR]
        assert entry["role"] == "overview"
        block = entry["zagg_overview"]
        assert block["spec"] == "zagg-overview/2"
        assert block["regime"] == "stage-merge"
        assert block["merges_from_raw"] == 2
        assert block["source_children"] == {"folded": 1, "missing": 0, "unreadable": 0}

    def test_rung_parity_with_the_column_partial(self, swept):
        # The rung's bytes ARE the leaf partial's, placed at the leaf's
        # nested rank: exact class equal, digest bytes decode identical.
        manifest = read_manifest(swept)
        rung = open_level(swept, 4)
        partial = open_column_order(swept, manifest, 4)
        assert int(rung["count"].values.sum()) == int(partial["count"].values.sum())
        element = parse_ragged_attrs(rung["h_tdigest"].attrs, field="h_tdigest")
        np.testing.assert_array_equal(
            decode_cell(rung["h_tdigest"].values[2], element),
            decode_cell(partial["h_tdigest"].values[0], element),
        )
        assert rung["h_tdigest"].values[0] == b""  # absent leaves keep the fill

    def test_shape_consistent_across_all_three_kinds(self, swept):
        # The model's one law, all three artifact kinds on one store.
        native = open_level(swept, 6)
        column = open_level(swept, 5)
        rung = open_level(swept, 4)
        for ds in (native, column, rung):
            assert list(ds.sizes) == ["cells"]
            assert set(ds.coords) == {"morton", "cell_ids"}
            assert isinstance(ds.xindexes["morton"], MortonMocIndex)
        assert set(rung.data_vars) == set(column.data_vars)
        assert set(column.data_vars) <= set(native.data_vars)

    def test_undeclared_cell_order_is_refused(self, swept):
        from moczarr.pyramid import open_overview_order

        manifest = read_manifest(swept)
        with pytest.raises(ValueError, match="strictly between"):
            open_overview_order(swept, manifest, 3, cell_order=7)

    def test_off_order_artifact_raises(self, swept, tmp_path):
        # Interpretable but wrong (§4.3 posture): the artifact's own
        # cell_order must be the level's — the cross-check that makes
        # cell_order= safe under either attrs revision.
        path = Path(swept) / "1" / "1" / "2" / "1" / "all.zarr" / "zarr.json"
        meta = json.loads(path.read_text())
        meta["attributes"]["zagg_overview"]["cell_order"] = 5
        path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="mis-rank"):
            open_level(swept, 4)

    def test_v1_default_path_degrades_on_a_conformant_v2_store(self, swept):
        # The pre-existing /1 entry point, called with no cell_order= on a
        # store whose ancestor artifact is perfectly conformant /2: the
        # constant-depth formula c-(s-k) names order 5 here while the ladder
        # entry says 4, and the artifact's level is the LADDER's. That
        # mismatch indicts the derivation, not the store, so the object is
        # skipped and the order degrades by omission (§4.1) — never the
        # "off-order objects would mis-rank rows" raise, which would make a
        # conformant store take the /1 reader down.
        from moczarr.pyramid import open_overview_order

        manifest = read_manifest(swept)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert open_overview_order(swept, manifest, 3) is None
        messages = [str(w.message) for w in caught]
        assert any("zagg-overview/2" in m and "cell_order=" in m for m in messages)
        assert any("no stamped overview object" in m for m in messages)

    def test_explicit_cell_order_still_raises_off_order(self, swept):
        # The other half of the same seam: when the CALLER names the level,
        # an artifact declaring another resolution is interpretable-but-wrong
        # on both attrs revisions and raises (the /1 half is pinned by
        # test_pyramid.py::test_off_order_overview_raises).
        from moczarr.pyramid import open_overview_order

        manifest = read_manifest(swept)
        with pytest.raises(ValueError, match="mis-rank"):
            open_overview_order(swept, manifest, 3, cell_order=5)

    def test_unswept_rung_degrades_to_none(self):
        # Declared-but-unmaterialized is a legal recorded state: the rung
        # at cells 3 (node 2) has no artifact on the unswept fixture.
        with pytest.warns(UserWarning, match="no stamped overview object"):
            assert open_level(TEMPORAL, 3) is None

    def test_demotions_ride_and_flatten(self, tmp_path):
        from moczarr import level_demotions

        root = tmp_path / "hive"
        shutil.copytree(TEMPORAL, root)
        recorded = [
            {
                "field": "composition",
                "class": "packed",
                "reason": "divisor-missing",
                "contributors": 1,
                "of": "h_tdigest",
                "cells": 4,
            }
        ]
        _write_stage_artifact(root, demotions=recorded)
        ds = open_level(str(root), 4)
        (entry,) = ds.attrs[OBJECTS_ATTR]
        assert entry["zagg_overview"]["demotions"] == recorded
        assert level_demotions(ds) == [{"node": "1121", "window": None, **recorded[0]}]

    def test_clean_level_has_no_demotions(self, swept):
        from moczarr import level_demotions

        assert level_demotions(open_level(swept, 4)) == []
        assert level_demotions(open_level(swept, 5)) == []

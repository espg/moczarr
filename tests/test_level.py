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

    def test_aoi_passes_through(self):
        full = open_level(self.ATL06, 6)
        word = np.uint64(full["morton"].values[0])
        cut = open_level(self.ATL06, 6, aoi=np.array([word], dtype=np.uint64))
        assert dict(cut.sizes) == {"cells": 1}
        assert int(cut["morton"].values[0]) == int(word)

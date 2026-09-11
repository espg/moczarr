"""Leaf column artifacts (zagg spec §4.6, ``zagg-column/1``) — issue #36.

Phase surface: the name seam stops *hiding* columns and starts *classifying*
them — ``walk_leaves`` still never yields one (a column is commit-stamped
like a leaf, so the completeness check downstream would wave it through),
while :func:`moczarr.store.walk_columns` discovers them as what they are.
Runs against the vendored spec §7 fixtures (``tests/data/spec/*``): real
zagg-writer bytes, each carrying one leaf plus its §4.6 column, declared
``zagg-pyramid/2`` and deliberately unswept (no overview artifacts).
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from moczarr import convention, store
from moczarr.column import column_orders, open_column, read_column_record
from moczarr.convention import column_name, column_path, is_column_basename, leaf_path

SPEC = Path(__file__).parent / "data" / "spec"
SERC = str(Path(__file__).parent / "data" / "serc_hive")
#: Every vendored spec fixture: one leaf, one sibling column at its node.
COLUMN_REL = "1/1/2/1/3/all.pyramid.zarr"
LEAF_REL = "1/1/2/1/3/11213.zarr"


class TestColumnNaming:
    def test_is_column_basename_is_the_suffix_seam(self):
        # §4.6: the suffix alone is normative — the frozen window charset
        # admits no ".", so no leaf or overview basename can end this way.
        assert is_column_basename("all.pyramid.zarr")
        assert is_column_basename("2019.pyramid.zarr")
        assert not is_column_basename("11213.zarr")
        assert not is_column_basename("all.zarr")
        assert not is_column_basename("11213_2019.zarr")
        assert not is_column_basename("all.pyramid.stats.json")

    def test_column_name_stems_from_the_window_alone(self):
        # §4.6 naming: `{window stem}.pyramid.zarr`, stem from the window
        # ALONE — the unwindowed case takes the reserved all-time token.
        assert column_name() == "all.pyramid.zarr"
        assert column_name("2019") == "2019.pyramid.zarr"

    def test_column_name_refuses_the_reserved_token_and_bad_labels(self):
        # window="all" is the same reserved-token refusal every window=
        # entry point shares (espg/moczarr#30); None is the spelling.
        with pytest.raises(ValueError, match="reserved all-time token"):
            column_name("all")
        with pytest.raises(ValueError, match="frozen grammar"):
            column_name("20_19")

    def test_column_path_is_the_leaf_sibling(self):
        # §4.6: one column per (node, window), a sibling of the leaf under
        # the leaf's own node prefix.
        assert column_path("11213") == "1/1/2/1/3/all.pyramid.zarr"
        assert column_path("11213", "2019") == "1/1/2/1/3/2019.pyramid.zarr"
        node = leaf_path("11213").rpartition("/")[0]
        assert column_path("11213").rpartition("/")[0] == node

    def test_column_path_grouped(self):
        # The grouped tree (spec §6.1) shares leaf_path's digit-chunking.
        assert column_path("11213", path_grouping=3) == "1/121/3/all.pyramid.zarr"
        node = leaf_path("11213", path_grouping=3).rpartition("/")[0]
        assert column_path("11213", path_grouping=3).rpartition("/")[0] == node

    def test_column_path_refuses_a_point_word(self):
        # Delegated shard validation: points never live in hive paths.
        word = convention.area29_to_point(convention.morton_word("11213" + "1" * 25))
        with pytest.raises(ValueError, match="POINT"):
            column_path(word)


class TestWalkColumns:
    @pytest.mark.parametrize("fixture", ["minimal", "temporal", "kitchen_sink"])
    def test_columns_discoverable_leaves_unpolluted(self, fixture):
        # The seam cuts both ways (issue #36): walk_leaves still never
        # yields the column — it is commit-stamped, so read_commit would
        # wave it through as a bona fide leaf — and walk_columns now
        # surfaces it as what it is.
        root = str(SPEC / fixture)
        assert list(store.walk_leaves(root)) == [LEAF_REL]
        assert list(store.walk_columns(root)) == [COLUMN_REL]
        # A column is stamped like a leaf (D4): completeness stays the
        # caller's check, shared with walk_leaves' contract.
        assert store.read_commit(root, COLUMN_REL)["complete"] is True

    def test_concurrent_walk_yields_the_same_set(self):
        root = str(SPEC / "minimal")
        assert sorted(store.walk_columns(root, concurrency=8)) == [COLUMN_REL]

    def test_store_without_columns_walks_empty(self):
        assert list(store.walk_columns(SERC)) == []

    def test_stage_columns_walk_but_are_not_addressable(self, tmp_path):
        # §4.6's issue-#384 stage columns are the same artifact shape at a
        # staged sweep's ANCESTOR dispatch nodes, and the suffix seam is the
        # seam at every depth — so the walk yields them beside the leaf
        # column, while the read pair (keyed by shard id) refuses them. The
        # split is on spec: stage-column existence at a given order is
        # orchestration, never contract, so a path-addressed reader is #36b.
        root = tmp_path / "hive"
        shutil.copytree(SPEC / "minimal", root)
        stage = root / "1" / "1" / "2"  # the order-2 ancestor of leaf 11213
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "all.pyramid.zarr").mkdir()
        payload = json.loads((root / COLUMN_REL / "zarr.json").read_text())
        (stage / "all.pyramid.zarr" / "zarr.json").write_text(json.dumps(payload))
        assert sorted(store.walk_columns(str(root))) == [
            COLUMN_REL,
            "1/1/2/all.pyramid.zarr",
        ]
        assert list(store.walk_leaves(str(root))) == [LEAF_REL]
        with pytest.raises(ValueError, match="LEAF columns"):
            read_column_record(str(root), "112")

    def test_windowed_column_names_walk_as_columns(self, tmp_path):
        # A windowed leaf's column is `{window}.pyramid.zarr` (§4.6); the
        # walker classifies by suffix, not stem.
        root = tmp_path / "hive"
        shutil.copytree(SPEC / "minimal", root)
        node = root / "1" / "1" / "2" / "1" / "3"
        (node / "2019.pyramid.zarr").mkdir()
        payload = json.loads((node / "all.pyramid.zarr" / "zarr.json").read_text())
        (node / "2019.pyramid.zarr" / "zarr.json").write_text(json.dumps(payload))
        assert sorted(store.walk_columns(str(root))) == [
            "1/1/2/1/3/2019.pyramid.zarr",
            COLUMN_REL,
        ]
        assert list(store.walk_leaves(str(root))) == [LEAF_REL]


class TestReadColumnRecord:
    MINIMAL = str(SPEC / "minimal")

    def test_minimal_record_binds(self):
        rec = read_column_record(self.MINIMAL, "11213")
        assert rec["spec"] == "zagg-column/1"
        assert rec["node"] == "11213"
        assert rec["order"] == 4
        assert rec["window"] == "all"
        assert sorted(rec["fields"]) == ["count", "h_tdigest"]
        assert rec["fields"]["h_tdigest"]["class"] == "approximate"
        # groups is the on-disk provenance map; column_orders spells it as
        # integers, finest first (the cells_with_data_order group leads).
        assert column_orders(rec) == (5, 4)
        assert rec["cells_with_data_order"] == 5
        assert all(g["regime"] == "leaf-column" for g in rec["groups"].values())

    def test_field_split_is_zero_open(self):
        # The §4.5 class split, read from the column alone — the published
        # ATL03 store's count-only vs four-field split (zagg#547) is this
        # same one-GET question. kitchen_sink declares every digest field
        # none-class, so its column materializes count only.
        count_only = read_column_record(str(SPEC / "kitchen_sink"), "11213")
        assert sorted(count_only["fields"]) == ["count"]
        multi = read_column_record(str(SPEC / "temporal"), "11213")
        assert sorted(multi["fields"]) == ["count", "h_tdigest"]

    def test_absent_column_reads_none(self):
        # §4.6: readers never require a column — absence is never an error.
        assert read_column_record(self.MINIMAL, "11212") is None

    def test_unstamped_column_is_debris(self, tmp_path):
        root = tmp_path / "hive"
        shutil.copytree(SPEC / "minimal", root)
        path = root / COLUMN_REL / "zarr.json"
        meta = json.loads(path.read_text())
        del meta["attributes"]["morton_hive_commit"]
        path.write_text(json.dumps(meta))
        assert read_column_record(str(root), "11213") is None

    @pytest.mark.parametrize(
        ("doctor", "match"),
        [
            (lambda a: a.__setitem__("role", "overview"), "not 'column'"),
            (lambda a: a.pop("zagg_column"), "lacks the 'zagg_column'"),
            (
                lambda a: a["zagg_column"].__setitem__("spec", "zagg-column/2"),
                "zagg-column/2",
            ),
            # The identity keys the record carries must be the ones asked
            # for: §4.6 makes `node` the leaf's morton decimal, `order` its
            # shard order, and `window` the key the basename round-trips
            # with. A column that disagrees would hand back another node's
            # cells under this shard's identity — the "interpretable but
            # wrong" class, not a missing one.
            (
                lambda a: a["zagg_column"].__setitem__("node", "11214"),
                "not this leaf's '11213'/4",
            ),
            (lambda a: a["zagg_column"].__setitem__("order", 3), "not this leaf's '11213'/4"),
            (lambda a: a["zagg_column"].__setitem__("window", "2019"), "round-trip"),
        ],
    )
    def test_stamped_nonconformant_column_raises(self, tmp_path, doctor, match):
        # Contrast the absent/unstamped cases: this call names ONE object,
        # and a stamped object at the column basename that does not classify
        # as a /1 column cannot be half-trusted (the conformance rule).
        root = tmp_path / "hive"
        shutil.copytree(SPEC / "minimal", root)
        path = root / COLUMN_REL / "zarr.json"
        meta = json.loads(path.read_text())
        doctor(meta["attributes"])
        path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match=match):
            read_column_record(str(root), "11213")

    def test_window_seam(self):
        # The same seam every window= entry point shares: an unwindowed
        # store refuses a label (and the reserved token), a windowed store
        # requires one — its columns are {window}.pyramid.zarr.
        with pytest.raises(ValueError, match="unwindowed stores"):
            read_column_record(self.MINIMAL, "11213", window="2019")
        with pytest.raises(ValueError, match="reserved all-time token"):
            read_column_record(self.MINIMAL, "11213", window="all")
        windowed = str(Path(__file__).parent / "data" / "overview_hive" / "atl06_windows")
        with pytest.raises(ValueError, match="pass window="):
            read_column_record(windowed, "4331244")
        # A windowed store without columns: absence, never an error.
        assert read_column_record(windowed, "4331244", window="2019") is None

    def test_cell_id_names_no_column(self):
        with pytest.raises(ValueError, match="order-5"):
            read_column_record(self.MINIMAL, "112131")


class TestColumnReads:
    """The normal ragged/dense read path, pointed at column groups."""

    MINIMAL = str(SPEC / "minimal")

    def test_dense_field_reads_and_folds(self):
        # count is exact-class (sum law): the node-order member (group 4)
        # and the declared resolution (group 5) both fold from the leaf's
        # resident cells, so all three totals agree — the §4.6 from-leaves
        # parity contract, checked through plain zarr opens.
        import zarr

        from moczarr import open_leaf

        col = open_column(self.MINIMAL, "11213")
        c5 = zarr.open_array(col, path="5/count", mode="r", zarr_format=3)[:]
        c4 = zarr.open_array(col, path="4/count", mode="r", zarr_format=3)[:]
        leaf = open_leaf(self.MINIMAL, "11213")
        c6 = zarr.open_array(leaf, path="6/count", mode="r", zarr_format=3)[:]
        assert c5.shape == (4,) and c4.shape == (1,)  # 4^(r - node) cells
        assert int(c5.sum()) == int(c4.sum()) == int(c6.sum()) > 0

    def test_group_morton_coordinates_are_node_descendants(self):
        import zarr

        col = open_column(self.MINIMAL, "11213")
        words5 = zarr.open_array(col, path="5/morton", mode="r", zarr_format=3)[:]
        assert [convention.morton_decimal(int(w))[:5] for w in words5] == ["11213"] * 4
        assert sorted(words5) == list(words5)  # ascending, §4.4 coordinate
        words4 = zarr.open_array(col, path="4/morton", mode="r", zarr_format=3)[:]
        assert [int(w) for w in words4] == [convention.morton_word("11213")]

    def test_ragged_digest_reads_through_read_ragged(self):
        # The SAME per-leaf ragged reader, pointed at a column group path —
        # no column-specific decode layer.
        from moczarr import read_ragged

        rows = list(read_ragged(open_column(self.MINIMAL, "11213"), "5/h_tdigest"))
        assert len(rows) == 3  # the stamp's cells_with_data, group 5
        for word, values in rows:
            assert convention.morton_decimal(int(word)).startswith("11213")
            assert values.ndim == 2 and values.shape[1] == 2  # (n, *inner_shape)

    def test_companion_channels_ride_along(self):
        # A column group carries every sibling its field's §4.5 entry
        # declares (locations AND times on the temporal fixture), exactly
        # as a leaf does — same reader, same flags.
        from moczarr import read_ragged

        col = open_column(str(SPEC / "temporal"), "11213")
        rows = list(read_ragged(col, "5/h_tdigest", locations=True, times=True))
        assert rows
        for _word, values, locs, times in rows:
            assert len(locs) == len(values) == len(times)


@pytest.mark.skipif(
    os.environ.get("MOCZARR_LIVE_TESTS", "").lower() not in {"1", "true", "yes"},
    reason="live S3 acceptance (anonymous, metadata + one tiny group); set MOCZARR_LIVE_TESTS=1",
)
class TestLiveAtl03Pyramid:
    """Issue #36 against the published ATL03 store (anonymous, cheap).

    Pins today's audited state (englacial/zagg#547, 2026-09-10): the
    v1-era spacing-2 [7,5,3,1] declaration, and the column split — ~204
    0.52-era four-field columns against ~2,713 count-only ones. Both halves
    move when the #547 runbook lands (dense /2 re-declaration; PR #524
    column backfill), so this test documents the shards it pins and will
    need the new facts then. Env-gated: the committed suite stays offline.
    """

    ROOT = "s3://us-west-2.opendata.source.coop/englacial/zagg/demo/atl03_tdigest_o9.zarr"
    S3 = {"region": "us-west-2", "anonymous": True}
    #: Audited examples of each column kind (probed 2026-09-10).
    COUNT_ONLY = "3213244424"
    FOUR_FIELD = "3133332144"

    def test_declaration_parse_and_presence_shape(self):
        from moczarr import read_pyramid

        # orders=[1] bounds the probe at the store's 6 order-1 ancestor
        # nodes — presence shape only, not a zero pin: the #547 sweep will
        # legitimately move stamped from 0.
        rp = read_pyramid(self.ROOT, orders=[1], **self.S3)
        decl = rp["declaration"]
        assert decl["spec"] == "zagg-pyramid/1"
        assert decl["orders"] == [7, 5, 3, 1]
        assert decl["spacing"] == 2
        assert decl["fold_source"] == "cascade"
        assert decl["cell_orders"][7] == [17]
        assert {f: e["class"] for f, e in decl["fields"].items()} == {
            "count": "exact",
            "h_tdigest_signal": "approximate",
            "h_tdigest_noise": "approximate",
            "composition": "packed",
        }
        (probe,) = rp["presence"].values()
        assert probe["nodes"] > 0 and 0 <= probe["stamped"] <= probe["nodes"]

    def test_column_split_and_groups(self):
        from moczarr import read_manifest

        manifest = read_manifest(self.ROOT, **self.S3)  # once, then threaded
        for shard, want in (
            (self.COUNT_ONLY, ["count"]),
            (
                self.FOUR_FIELD,
                ["composition", "count", "h_tdigest_noise", "h_tdigest_signal"],
            ),
        ):
            rec = read_column_record(self.ROOT, shard, manifest=manifest, **self.S3)
            assert sorted(rec["fields"]) == want
            assert column_orders(rec) == (13, 12, 11, 10, 9)

    def test_column_group_reads_through_the_normal_path(self):
        import zarr

        from moczarr import read_ragged

        col = open_column(self.ROOT, self.FOUR_FIELD, **self.S3)
        # The node-order member: one cell, the leaf's whole-footprint
        # aggregate — the cheapest real read on the store.
        count9 = zarr.open_array(col, path="9/count", mode="r", zarr_format=3)[:]
        assert count9.shape == (1,) and int(count9[0]) > 0
        rows = list(read_ragged(col, "9/h_tdigest_signal"))
        assert len(rows) == 1
        word, values = rows[0]
        assert convention.morton_decimal(int(word)) == self.FOUR_FIELD
        assert values.shape[1] == 2

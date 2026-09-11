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
import shutil
from pathlib import Path

import pytest

from moczarr import convention, store
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

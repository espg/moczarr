"""Convention core: ids, hive paths, leaf names, manifest validation.

Golden vectors are pinned against zagg's writer (``zagg.hive`` /
``zagg.windows`` test suites) so the read and write implementations of the
mortie#62 convention cannot drift silently.
"""

import numpy as np
import pytest

from moczarr import convention

# Order-6 southern shard used across zagg's hive tests (and its northern
# mirror: the string arithmetic is sign-dependent, so both hemispheres run).
SHARD = "-5112333"
NORTH = "5112333"
#: Golden packed word for SHARD (pinned against mortie 0.9.0).
SHARD_WORD = 12711972898206646278


@pytest.fixture(params=[SHARD, NORTH])
def shard(request):
    return request.param


class TestIds:
    def test_word_decimal_round_trip(self, shard):
        word = convention.morton_word(shard)
        assert convention.morton_decimal(word) == shard
        # Pass-throughs: str in -> str out, int in -> int out.
        assert convention.morton_decimal(shard) == shard
        assert convention.morton_word(word) == word

    def test_golden_word(self):
        assert convention.morton_word(SHARD) == SHARD_WORD

    def test_order_base_rank(self, shard):
        assert convention.decimal_order(shard) == 6
        assert convention.decimal_base(shard) == ("-5" if shard.startswith("-") else "5")
        # Tail 112333 -> base-4 digits 001222 -> rank.
        expected = int("001222", 4)
        assert convention.decimal_rank(shard) == expected
        tail = shard[len(convention.decimal_base(shard)) :]
        assert convention.rank_tail(expected, 6) == tail

    def test_rank_tail_round_trip(self):
        for rank in range(4**3):
            tail = convention.rank_tail(rank, 3)
            assert len(tail) == 3 and set(tail) <= set("1234")
            assert convention.decimal_rank("1" + tail) == rank

    def test_is_base_component(self):
        assert convention.is_base_component("5")
        assert convention.is_base_component("-5")
        assert not convention.is_base_component("7")
        assert not convention.is_base_component("55")
        assert not convention.is_base_component("morton_hive.json")


class TestLeafPath:
    def test_golden_path(self):
        assert convention.leaf_path(SHARD) == "-5/1/1/2/3/3/3/-5112333.zarr"
        assert convention.leaf_path(NORTH) == "5/1/1/2/3/3/3/5112333.zarr"

    def test_word_input(self):
        assert convention.leaf_path(SHARD_WORD) == convention.leaf_path(SHARD)

    def test_windowed_leaf(self, shard):
        path = convention.leaf_path(shard, window="2019")
        assert path.endswith(f"/{shard}_2019.zarr")
        assert path.rsplit("/", 1)[0] == convention.leaf_path(shard).rsplit("/", 1)[0]

    def test_bad_window_label_rejected(self, shard):
        with pytest.raises(ValueError, match="frozen grammar"):
            convention.leaf_path(shard, window="20_19")

    def test_negative_int_rejected(self):
        # A decimal id read as a signed int (the natural user mistake) must
        # fail with an actionable ValueError, not a bare numpy OverflowError.
        with pytest.raises(ValueError, match="packed morton word"):
            convention.leaf_path(-5112333)


class TestLeafNames:
    def test_split_round_trip(self):
        assert convention.split_leaf_name("-5112333.zarr") == ("-5112333", None)
        assert convention.split_leaf_name("-5112333_2019.zarr") == ("-5112333", "2019")
        assert convention.leaf_name("-5112333", "2019") == "-5112333_2019.zarr"
        assert convention.leaf_name("-5112333") == "-5112333.zarr"

    def test_first_underscore_splits(self):
        # Labels cannot contain "_", so a second underscore is malformed.
        with pytest.raises(ValueError, match="frozen grammar"):
            convention.split_leaf_name("-5112333_20_19.zarr")

    def test_non_zarr_rejected(self):
        with pytest.raises(ValueError, match="not a leaf zarr name"):
            convention.split_leaf_name("morton_hive.json")

    def test_label_grammar(self):
        assert convention.validate_label("2019") == "2019"
        assert convention.validate_label("2019-07") == "2019-07"
        for bad in ("", "a" * 33, "20_19", "2019!", None):
            with pytest.raises(ValueError):
                convention.validate_label(bad)


class TestNodeInvariant:
    def test_golden_path_passes(self):
        convention.check_node_invariant("-5/1/1/2/3/3/3/-5112333.zarr")
        convention.check_node_invariant("-5/1/1/2/3/3/3/-5112333_2019.zarr")

    @pytest.mark.parametrize(
        "bad",
        [
            "-5112333.zarr",  # no digit components
            "-5/1/1/2/3/3/3/-5112334.zarr",  # id != concatenated components
            "-5/1/1/2/3/3/5/-5112335.zarr",  # digit outside 1..4
            "-7/1/-71.zarr",  # base outside 1..6
            "-5/1/1/2/3/3/3/-5112333",  # not a .zarr leaf
            "-5/11/2/3/3/3/-5112333.zarr",  # grouped digits
            "-5/1/1/2/3/3/3/-5112333_20_19.zarr",  # malformed window label
        ],
    )
    def test_violations_raise(self, bad):
        with pytest.raises(ValueError, match="node invariant"):
            convention.check_node_invariant(bad)


# Order-29 decimal strings (both hemispheres) and their golden packed words
# — area (unmarked, the §4 tie-break) and point (p-marked) forms, literals
# pinned once against mortie 0.9.0 + the spec §1 suffix arithmetic.
AREA29_NORTH = "4" + "1234" * 7 + "2"
AREA29_SOUTH = "-5" + "4321" * 7 + "1"
AREA29_NORTH_WORD = 4733760060091642285  # suffix 45
POINT_NORTH_WORD = 4733760060091642301  # suffix 61
AREA29_SOUTH_WORD = 13712984013617909341  # suffix 29
POINT_SOUTH_WORD = 13712984013617909360  # suffix 48


class TestPointKind:
    """Spec §1/§4: kind is encoding-carried (suffix band), never metadata."""

    def test_suffix_table_bands(self):
        # §1 table, incl. the 27/28 and 47/48 band boundaries, both
        # hemispheres: 0..=27 area (order == suffix), 28..=47 order-28/29
        # area preorder, 48..=63 order-29 point.
        for base in ("4", "-5"):
            o27 = convention.morton_word(base + "1" * 27)
            o28 = convention.morton_word(base + "1" * 28)
            area_lo = convention.morton_word(base + "1" * 29)  # suffix 29
            area_hi = convention.morton_word(base + "1" * 27 + "44")  # suffix 47
            assert o27 & 0x3F == 27 and not convention.is_point_word(o27)
            assert o28 & 0x3F == 28 and not convention.is_point_word(o28)
            assert area_lo & 0x3F == 29 and not convention.is_point_word(area_lo)
            assert area_hi & 0x3F == 47 and not convention.is_point_word(area_hi)
            point_lo = convention.area29_to_point(area_lo)
            point_hi = convention.area29_to_point(area_hi)
            assert point_lo & 0x3F == 48 and convention.is_point_word(point_lo)
            assert point_hi & 0x3F == 63 and convention.is_point_word(point_hi)
        # Vectorized form: bool array over mixed kinds.
        mixed = np.asarray([AREA29_NORTH_WORD, POINT_NORTH_WORD], dtype=np.uint64)
        np.testing.assert_array_equal(convention.is_point_word(mixed), [False, True])

    def test_p_round_trip_goldens(self):
        # BOTH §4 parse directions, both hemispheres: p-marked -> POINT word,
        # unmarked order-29 -> AREA word (the tie-break); renders invert both.
        for dec, area_word, point_word in (
            (AREA29_NORTH, AREA29_NORTH_WORD, POINT_NORTH_WORD),
            (AREA29_SOUTH, AREA29_SOUTH_WORD, POINT_SOUTH_WORD),
        ):
            assert convention.morton_word(dec) == area_word
            assert convention.morton_word(dec + "p") == point_word
            assert convention.morton_decimal(area_word) == dec
            assert convention.morton_decimal(point_word) == dec + "p"

    def test_p_legal_only_at_order29(self):
        for bad in ("41p", "4" + "1" * 28 + "p", "p"):
            with pytest.raises(ValueError, match="order-29"):
                convention.morton_word(bad)

    def test_twin_round_trips(self):
        for area, point in (
            (AREA29_NORTH_WORD, POINT_NORTH_WORD),
            (AREA29_SOUTH_WORD, POINT_SOUTH_WORD),
        ):
            assert convention.area29_to_point(area) == point
            assert convention.point_to_area29(point) == area
        # Area words pass through the point->area normalization unchanged.
        assert convention.point_to_area29(AREA29_NORTH_WORD) == AREA29_NORTH_WORD
        # Sub-29 words have no point twin (points exist only at order 29).
        for bad in (convention.morton_word("41"), convention.morton_word("4" + "1" * 28)):
            with pytest.raises(ValueError, match="order-29"):
                convention.area29_to_point(bad)

    def test_paths_never_carry_points(self):
        # leaf_path rejects point words and p-marked ids (spec §2/§6.6)...
        with pytest.raises(ValueError, match="POINT"):
            convention.leaf_path(POINT_NORTH_WORD)
        with pytest.raises(ValueError, match="POINT"):
            convention.leaf_path(AREA29_SOUTH + "p")
        # ...and the node invariant rejects a p-suffixed leaf id.
        with pytest.raises(ValueError, match="kind-suffix"):
            convention.check_node_invariant("4/1/41p.zarr")
        # The unmarked order-29 path stays legal (parses as AREA, §4).
        rel = convention.leaf_path(AREA29_NORTH)
        convention.check_node_invariant(rel)


class TestPathGrouping:
    """The D21 digit-chunking (spec §6.1): grouping is the ONE path code path."""

    def test_group_digits(self):
        assert convention.group_digits("112333", 1) == list("112333")
        assert convention.group_digits("112333", 3) == ["112", "333"]
        # The LAST component carries the remainder (leading stay full-width).
        assert convention.group_digits("33142241", 3) == ["331", "422", "41"]
        assert convention.group_digits("", 3) == []

    def test_leaf_path_grouped_goldens(self):
        # Both hemispheres + the short remainder component (8 % 3 == 2).
        assert convention.leaf_path("433142241", path_grouping=3) == "4/331/422/41/433142241.zarr"
        assert (
            convention.leaf_path("-433412214", path_grouping=3) == "-4/334/122/14/-433412214.zarr"
        )
        # Evenly dividing order: no short component.
        assert convention.leaf_path("-5112333", path_grouping=3) == "-5/112/333/-5112333.zarr"
        assert convention.leaf_path("-5112333", path_grouping=6) == "-5/112333/-5112333.zarr"

    def test_grouping_one_is_the_same_path(self):
        # path_grouping=1 must be byte-identical to the (mortie-delegated)
        # default — one generic chunking, never a separate branch.
        assert convention.leaf_path(SHARD, path_grouping=1) == convention.leaf_path(SHARD)

    def test_windowed_grouped(self):
        path = convention.leaf_path("-5112333", window="2019", path_grouping=3)
        assert path == "-5/112/333/-5112333_2019.zarr"

    def test_node_invariant_grouped(self):
        convention.check_node_invariant("4/331/422/41/433142241.zarr", path_grouping=3)
        convention.check_node_invariant("-5/112/333/-5112333_2019.zarr", path_grouping=3)
        for bad in (
            "-5/1/1/2/3/3/3/-5112333.zarr",  # one-digit components under grouping 3
            "-5/11/23/33/-5112333.zarr",  # short NON-terminal component
            "4/331/422/414/43314224.zarr",  # id != concatenated components
        ):
            with pytest.raises(ValueError, match="node invariant"):
                convention.check_node_invariant(bad, path_grouping=3)
        # A grouped path is a violation under grouping 1 (and the default).
        with pytest.raises(ValueError, match="node invariant"):
            convention.check_node_invariant("-5/112/333/-5112333.zarr")

    def test_manifest_accessor(self):
        assert convention.manifest_path_grouping({"path_grouping": 3}) == 3
        assert convention.manifest_path_grouping({}) == 1  # D21: absent -> 1


def _manifest(**overrides):
    base = {
        "spec": convention.HIVE_SPEC,
        "dataset": {"short_name": "ATL06", "version": "007"},
        "cell_order": 19,
        "shard_order": 9,
        "split_schedule": [1] * 9,
        "pyramid": {"orders": [], "aggregation": {}},
        "generated_at": "2026-07-17T00:00:00+00:00",
    }
    base.update(overrides)
    return base


class TestManifest:
    def test_v1_parses(self):
        assert convention.parse_manifest(_manifest()) == _manifest()

    def test_v2_parses(self):
        payload = _manifest(
            spec=convention.HIVE_SPEC_V2,
            temporal={"schedule": "yearly", "time_field": "delta_time"},
        )
        assert convention.parse_manifest(payload) == payload

    def test_unknown_spec_rejected(self):
        with pytest.raises(ValueError, match="unknown manifest spec"):
            convention.parse_manifest(_manifest(spec="morton-hive/9"))

    def test_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="not a mapping"):
            convention.parse_manifest([1, 2])

    def test_missing_order_rejected(self):
        bad = _manifest()
        del bad["cell_order"]
        with pytest.raises(ValueError, match="cell_order"):
            convention.parse_manifest(bad)

    def test_inverted_orders_rejected(self):
        with pytest.raises(ValueError, match="cells nest inside shards"):
            convention.parse_manifest(_manifest(cell_order=5))

    def test_v2_requires_temporal(self):
        with pytest.raises(ValueError, match="temporal block"):
            convention.parse_manifest(_manifest(spec=convention.HIVE_SPEC_V2))

    def test_path_grouping_validated(self):
        assert convention.parse_manifest(_manifest(path_grouping=3))["path_grouping"] == 3
        assert "path_grouping" not in convention.parse_manifest(_manifest())  # absent ok
        for bad in (0, -1, "3", [1, 2], True, None):
            with pytest.raises(ValueError, match="path_grouping"):
                convention.parse_manifest(_manifest(path_grouping=bad))

    def test_v1_refuses_temporal(self):
        with pytest.raises(ValueError, match="must not carry"):
            convention.parse_manifest(_manifest(temporal={"schedule": "yearly"}))


def test_words_are_uint64_scale():
    # Packed words exceed 2^53: the reason range endpoints are strings.
    assert convention.morton_word(SHARD) > 2**53
    assert np.uint64(convention.morton_word(SHARD)) == SHARD_WORD

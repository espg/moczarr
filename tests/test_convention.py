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

    def test_v1_refuses_temporal(self):
        with pytest.raises(ValueError, match="must not carry"):
            convention.parse_manifest(_manifest(temporal={"schedule": "yearly"}))


def test_words_are_uint64_scale():
    # Packed words exceed 2^53: the reason range endpoints are strings.
    assert convention.morton_word(SHARD) > 2**53
    assert np.uint64(convention.morton_word(SHARD)) == SHARD_WORD

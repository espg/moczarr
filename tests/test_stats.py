"""D20 stats sidecars, D22 rollups, and the O11 verifier, against the goldens.

The committed ``tests/data/multiproduct_hive`` fixture carries the full
telemetry surface: ``atl06`` has ``stats.json`` sidecars (with O11
``content_hashes`` computed independently by the generator — the verifier is
golden-tested, not self-tested) and hand-folded ``stats.rollup.json``
objects at the shard nodes and every ancestor; ``atl06_windows`` has
``stats_{window}.json`` sidecars in all three states (with hashes, without,
absent). Name arithmetic is pinned across all three spec grammars,
including the D23 ``{window}.stats.json`` / ``all`` form moczarr will meet
on ``/3`` stores.
"""

import json
import shutil
from pathlib import Path

import pytest

from moczarr import convention, stats
from moczarr.stats import (
    combined_hash,
    hash_arrays,
    read_stats,
    read_stats_rollup,
    stats_sidecar_key,
    stats_sidecar_path,
    verify_arrays,
)

FIXTURE = Path(__file__).parent / "data" / "multiproduct_hive"


@pytest.fixture()
def atl06():
    return str(FIXTURE / "atl06")


@pytest.fixture()
def windows():
    return str(FIXTURE / "atl06_windows")


class TestSidecarNaming:
    def test_legacy_bare(self):
        for spec in (None, convention.HIVE_SPEC, convention.HIVE_SPEC_V2):
            assert stats_sidecar_key("4111.zarr", spec) == "stats.json"

    def test_legacy_windowed(self):
        assert stats_sidecar_key("-5111_2019.zarr", convention.HIVE_SPEC_V2) == "stats_2019.json"

    def test_v3_window_and_all_token(self):
        # D23: sidecar = leaf stem + .stats.json; `all` is the schedule-none
        # reserved token and satisfies the label grammar by design.
        assert stats_sidecar_key("2019.zarr", convention.HIVE_SPEC_V3) == "2019.stats.json"
        assert stats_sidecar_key("all.zarr", convention.HIVE_SPEC_V3) == "all.stats.json"

    def test_unknown_spec_raises(self):
        # A spec bump must fail loudly, never key the legacy name.
        with pytest.raises(ValueError, match="unknown store spec"):
            stats_sidecar_key("4111.zarr", "morton-hive/9")

    def test_malformed_names_raise(self):
        with pytest.raises(ValueError):
            stats_sidecar_key("2019", convention.HIVE_SPEC_V3)  # not a .zarr
        with pytest.raises(ValueError):
            stats_sidecar_key("a/b.zarr", convention.HIVE_SPEC_V3)  # path escape
        with pytest.raises(ValueError):
            stats_sidecar_key("41_bad_label!.zarr", convention.HIVE_SPEC)

    def test_sidecar_path_is_sibling(self):
        rel = convention.leaf_path("4111")
        assert stats_sidecar_path(rel, convention.HIVE_SPEC) == "4/1/1/1/stats.json"


class TestReadStats:
    def test_reads_leaf_record(self, atl06):
        record = read_stats(atl06, "4111")
        assert record["schema_version"] == 1
        assert record["shard_key"] == convention.morton_word("4111")
        assert record["n_obs"] == 100
        assert record["semantic_hash"] is not None

    def test_accepts_packed_word(self, atl06):
        word = convention.morton_word("4112")
        assert read_stats(atl06, word)["n_obs"] == 200

    def test_windowed_sidecars(self, windows):
        assert read_stats(windows, "-5111", window="2019")["n_obs"] == 10
        assert read_stats(windows, "-5111", window="2020")["n_obs"] == 20

    def test_absent_reads_none(self, windows, atl06):
        assert read_stats(windows, "-5112", window="2019") is None  # leaf, no sidecar
        assert read_stats(atl06, "4113") is None  # no leaf at all

    def test_malformed_sidecar_reads_none(self, atl06, tmp_path):
        # Telemetry posture (D9 class): garbage degrades to absent, never raises.
        root = tmp_path / "store"
        shutil.copytree(atl06, root)
        (root / "4" / "1" / "1" / "1" / "stats.json").write_text("{not json")
        assert read_stats(str(root), "4111") is None

    def test_no_manifest_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a hive store root"):
            read_stats(str(tmp_path), "4111")


class TestReadStatsRollup:
    def test_shard_node_envelope(self, atl06):
        envelope = read_stats_rollup(atl06, "4111")
        assert envelope["spec"] == stats.SWEEP_SPEC
        assert (envelope["family"], envelope["node"], envelope["order"]) == ("stats", "4111", 3)
        assert envelope["windows"] == [None]
        assert envelope["generation"]["n_leaves"] == 1
        assert envelope["payload"]["n_obs"] == 100

    def test_interior_fold_equals_direct(self, atl06):
        # rollup == direct (the D22 standing obligation): the interior
        # payload sums the leaf records, identity fields collapse per the
        # D20 merge dispositions.
        leaf = [read_stats(atl06, s) for s in ("4111", "4112")]
        for node in ("411", "41", "4"):
            envelope = read_stats_rollup(atl06, node)
            payload = envelope["payload"]
            assert envelope["generation"]["n_leaves"] == 2
            assert payload["n_obs"] == sum(r["n_obs"] for r in leaf)
            assert payload["duration_s"] == sum(r["duration_s"] for r in leaf)
            assert payload["shard_key"] is None  # mismatch -> None (absorbing)
            assert payload["run_id"] == leaf[0]["run_id"]  # common -> kept
            assert "content_hashes" not in payload  # per-leaf identity never folds

    def test_absent_and_unusable_read_none(self, atl06, windows, tmp_path):
        assert read_stats_rollup(windows, "-511") is None  # never swept
        root = tmp_path / "store"
        shutil.copytree(atl06, root)
        target = root / "4" / "1" / "stats.rollup.json"
        target.write_text(json.dumps({"spec": "other/1", "payload": {}}))
        assert read_stats_rollup(str(root), "41") is None  # wrong spec: cache posture


class TestVerifyArrays:
    def test_golden_match(self, atl06):
        # The fixture's recorded hashes were computed independently by the
        # generator from the written chunk bytes.
        result = verify_arrays(atl06, "4111")
        assert result["match"] is True
        assert result["mismatched"] == []
        assert set(result["computed"]) == {"5/morton", "5/count"}
        assert result["recorded"] == result["computed"]
        assert result["recorded_combined"] == result["combined"]

    def test_windowed_golden_match(self, windows):
        result = verify_arrays(windows, "-5111", window="2019")
        assert result["match"] is True
        assert set(result["computed"]) == {"5/morton", "5/height"}

    def test_tamper_localizes_mismatch(self, atl06, tmp_path):
        # O11's mismatch-localizer job: flip one value in one array and only
        # that array (plus the combined hash) reports different.
        root = tmp_path / "store"
        shutil.copytree(atl06, root)
        chunk = root / "4" / "1" / "1" / "1" / "4111.zarr" / "5" / "count" / "c" / "0"
        data = bytearray(chunk.read_bytes())
        data[0] ^= 0xFF
        chunk.write_bytes(bytes(data))
        result = verify_arrays(str(root), "4111")
        assert result["match"] is False
        assert result["mismatched"] == ["5/count"]
        assert result["combined"] != result["recorded_combined"]

    def test_nothing_recorded_is_none(self, windows):
        # A sidecar without content_hashes (and an absent sidecar) verify as
        # None — "unverifiable", distinct from both True and False.
        without = verify_arrays(windows, "-5111", window="2020")
        assert without["match"] is None and without["recorded"] is None
        absent = verify_arrays(windows, "-5112", window="2019")
        assert absent["match"] is None
        assert set(absent["computed"]) == {"5/morton", "5/height"}

    def test_flat_recorded_shape_accepted(self, atl06, tmp_path):
        # The O11 wording admits a flat {name: hash} record; the verifier
        # accepts it so whichever shape zagg's writer lands keeps verifying.
        root = tmp_path / "store"
        shutil.copytree(atl06, root)
        sidecar = root / "4" / "1" / "1" / "1" / "stats.json"
        record = json.loads(sidecar.read_text())
        record["content_hashes"] = dict(record["content_hashes"]["arrays"])
        sidecar.write_text(json.dumps(record))
        result = verify_arrays(str(root), "4111")
        assert result["match"] is True
        assert result["recorded_combined"] is None

    def test_combined_hash_is_order_free(self):
        a = {"x": "aa", "y": "bb"}
        b = {"y": "bb", "x": "aa"}
        assert combined_hash(a) == combined_hash(b)

    def test_hash_arrays_decoded_values(self, atl06):
        # Hashes are over decoded values (raw little-endian C-order bytes at
        # the declared dtype) — recomputable from the arrays themselves.
        import hashlib

        import numpy as np

        rel = convention.leaf_path("4111")
        hashes = hash_arrays(atl06, rel)
        count = np.arange(1, 17, dtype="<i8")
        assert hashes["5/count"] == hashlib.sha256(count.tobytes()).hexdigest()

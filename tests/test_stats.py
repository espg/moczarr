"""D20 stats sidecars, D22 rollups, and the O11 verifier, against the goldens.

The committed ``tests/data/multiproduct_hive`` fixture carries the full
telemetry surface: ``atl06`` has ``stats.json`` sidecars (with O11 per-array
``content_hashes`` computed independently by the generator from the written
chunk bytes — the verifier is golden-tested, not self-tested; the COMBINED
digest, which the generator does take from ``combined_hash``, is pinned by
the :data:`FROZEN_COMBINED_4111` literal instead) and hand-folded
``stats.rollup.json`` objects at the shard nodes and every ancestor;
``atl06_windows`` has ``stats_{window}.json`` sidecars in all three states
(with hashes, without,
absent); ``atl06_ragged`` carries the vlen (ragged) O11 surface — a
``variable_length_bytes`` digest array plus its ``{field}_locations``
sibling — so the length-prefixed vlen hash recipe is golden-pinned; and
``atl06_pg3`` is a ``path_grouping: 3`` store, so the D21 grouped sidecar and
rollup key arithmetic is exercised rather than only supported. Name
arithmetic is pinned across all three spec grammars, including the D23
``{window}.stats.json`` / ``all`` form moczarr will meet on ``/3`` stores.
"""

import json
import shutil
from pathlib import Path

import pytest

from moczarr import convention, stats
from moczarr.stats import (
    combined_hash,
    hash_arrays,
    overview_sidecar_key,
    overview_sidecar_path,
    read_overview_order_stats,
    read_overview_stats,
    read_stats,
    read_stats_rollup,
    stats_sidecar_key,
    stats_sidecar_path,
    verify_arrays,
    verify_overview_arrays,
)

FIXTURE = Path(__file__).parent / "data" / "multiproduct_hive"
#: The zagg-written overview-pyramid fixture (PR #28, regenerated at the zagg
#: sha that writes overview D20 sidecars — englacial/zagg PR #356).
OVERVIEW = Path(__file__).parent / "data" / "overview_hive"
OVERVIEW_GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "overview_hive.golden.json").read_text()
)

#: The FROZEN O11 combined-hash serialization, as a literal: sha256 of the
#: sorted per-array hex digests, ``"\n"``-joined, ASCII — for the ``atl06``
#: fixture's ``4111`` leaf. A literal because both sides of the fixture's
#: golden run through ``combined_hash`` (the generator records what the
#: verifier recomputes), so changing the joiner, hashing raw digest bytes, or
#: including array names would otherwise pass unnoticed — and this is the one
#: O11 choice this PR asks zagg's future writer to adopt verbatim.
FROZEN_COMBINED_4111 = "a5d44b9f2b35478e2f9d763c52146b14c8ab8934af64d40c4efa6400bf1c6670"


@pytest.fixture()
def atl06():
    return str(FIXTURE / "atl06")


@pytest.fixture()
def windows():
    return str(FIXTURE / "atl06_windows")


@pytest.fixture()
def ragged():
    return str(FIXTURE / "atl06_ragged")


@pytest.fixture()
def pg3():
    return str(FIXTURE / "atl06_pg3")


@pytest.fixture()
def ov_flat():
    """The unwindowed overview product — a ``morton-hive/1`` store."""
    return str(OVERVIEW / "atl06")


@pytest.fixture()
def ov_windows():
    """The windowed (``morton-hive/2``) overview product."""
    return str(OVERVIEW / "atl06_windows")


def _golden_stats(product: str, order: str, key: str) -> dict:
    """The golden's recorded sidecar entries for one order/window."""
    return OVERVIEW_GOLDEN["products"][product]["overviews"][order][key]["stats"]


def _rewrite_content_hashes(atl06, tmp_path, rewrite):
    """Copy the atl06 product, reshaping leaf ``4111``'s recorded hashes."""
    root = tmp_path / "store"
    shutil.copytree(atl06, root)
    sidecar = root / "4" / "1" / "1" / "1" / "stats.json"
    record = json.loads(sidecar.read_text())
    record["content_hashes"] = rewrite(record["content_hashes"])
    sidecar.write_text(json.dumps(record))
    return root


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

    def test_unstamped_generation_reads_none(self, atl06, tmp_path):
        # Same depth as the sweep's own reader: n_leaves must be an int, so a
        # `{"n_leaves": "2"}` stamp reads as absent here exactly as it would
        # be rejected writer-side.
        root = tmp_path / "store"
        shutil.copytree(atl06, root)
        target = root / "4" / "1" / "stats.rollup.json"
        envelope = json.loads(target.read_text())
        envelope["generation"]["n_leaves"] = "2"
        target.write_text(json.dumps(envelope))
        assert read_stats_rollup(str(root), "41") is None


class TestPathGrouping:
    """The D21 generic path at ``path_grouping: 3`` (espg/moczarr#11 directive:
    exercised, not theoretically supported). ``atl06_pg3`` is an order-5 shard
    whose digit tail chunks 3+2, so both a full-width component and the short
    remainder are covered.
    """

    def test_node_rel_chunks_the_tail(self):
        # The remainder rides LAST — easy to get backwards.
        assert stats._node_rel("411121", 3) == "4/111/21"
        assert stats._node_rel("41111", 3) == "4/111/1"  # 4-digit tail: 3 + 1
        assert stats._node_rel("4111", 3) == "4/111"  # exactly one component
        assert stats._node_rel("4", 3) == "4"
        assert stats._node_rel("-5111211", 3) == "-5/111/211"
        assert stats._node_rel("411121", 1) == "4/1/1/1/2/1"

    def test_sidecar_path_is_grouped(self):
        leaf = convention.leaf_path("411121", path_grouping=3)
        assert stats_sidecar_path(leaf, convention.HIVE_SPEC) == "4/111/21/stats.json"

    def test_reads_grouped_sidecar(self, pg3):
        record = read_stats(pg3, "411121")
        assert record["n_obs"] == 40
        assert record["shard_key"] == convention.morton_word("411121")

    def test_reads_grouped_rollups(self, pg3):
        # Component-boundary nodes are the grouped tree's real levels.
        for node in ("411121", "4111", "4"):
            envelope = read_stats_rollup(pg3, node)
            assert envelope["payload"]["n_obs"] == 40
            assert envelope["generation"]["n_leaves"] == 1
        assert read_stats_rollup(pg3, "41112") is None  # mid-component: no node

    def test_verifies_grouped_leaf(self, pg3):
        result = verify_arrays(pg3, "411121")
        assert result["match"] is True
        assert set(result["computed"]) == {"6/morton", "6/count"}


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
        root = _rewrite_content_hashes(atl06, tmp_path, lambda content: dict(content["arrays"]))
        result = verify_arrays(str(root), "4111")
        assert result["match"] is True
        assert result["recorded_combined"] is None

    def test_flat_shape_with_combined_key_accepted(self, atl06, tmp_path):
        # The likeliest flat encoding puts arrays and the combined hash in ONE
        # mapping; `combined` is reserved, never read as a phantom array name
        # (which reported this intact leaf as mismatched on 'combined').
        root = _rewrite_content_hashes(
            atl06,
            tmp_path,
            lambda content: {**content["arrays"], "combined": content["combined"]},
        )
        result = verify_arrays(str(root), "4111")
        assert result["match"] is True
        assert result["mismatched"] == []
        assert result["recorded_combined"] == result["combined"]

    def test_flat_combined_only_is_unverifiable(self, atl06, tmp_path):
        # The degenerate combined-only record records no per-array hashes, so
        # it is unverifiable — not "every array mismatched".
        root = _rewrite_content_hashes(atl06, tmp_path, lambda content: {"combined": "x"})
        result = verify_arrays(str(root), "4111")
        assert result["match"] is None
        assert result["recorded"] is None
        assert result["mismatched"] == []

    def test_combined_disagreement_is_never_a_match(self, atl06, tmp_path):
        # Per-array hashes all matching but the combined hash differing is
        # writer-serialization drift, not tampering — a distinct signal, and
        # never reported as verified.
        root = _rewrite_content_hashes(
            atl06, tmp_path, lambda content: {**content, "combined": "dead" * 16}
        )
        result = verify_arrays(str(root), "4111")
        assert result["mismatched"] == []
        assert result["combined_match"] is False
        assert result["match"] is False

    def test_combined_match_is_none_when_unrecorded(self, atl06, tmp_path):
        # A record with per-array hashes but no combined: verified per-array,
        # combined unverifiable.
        root = _rewrite_content_hashes(
            atl06, tmp_path, lambda content: {"arrays": content["arrays"]}
        )
        result = verify_arrays(str(root), "4111")
        assert result["match"] is True
        assert result["combined_match"] is None

    def test_empty_arrays_mapping_is_unverifiable(self, atl06, tmp_path):
        # {"arrays": {}} records nothing; reporting every array as mismatched
        # would read "nothing recorded" as "tampered".
        root = _rewrite_content_hashes(atl06, tmp_path, lambda content: {"arrays": {}})
        result = verify_arrays(str(root), "4111")
        assert result["match"] is None
        assert result["recorded"] is None
        assert result["mismatched"] == []

    def test_combined_serialization_is_frozen(self, atl06):
        # Not self-tested: the literal is the pin, and the recipe is
        # hand-rolled here rather than borrowed from combined_hash.
        import hashlib

        result = verify_arrays(atl06, "4111")
        assert result["combined"] == FROZEN_COMBINED_4111
        assert result["recorded_combined"] == FROZEN_COMBINED_4111
        digests = sorted(result["recorded"].values())
        hand_rolled = hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()
        assert hand_rolled == FROZEN_COMBINED_4111

    def test_combined_hash_is_order_free(self):
        a = {"x": "aa", "y": "bb"}
        b = {"y": "bb", "x": "aa"}
        assert combined_hash(a) == combined_hash(b)

    def test_vlen_golden_match(self, ragged):
        # The ragged (vlen-bytes) arrays are in O11's scope; the naive
        # `values.tobytes()` on an object array hashes POINTER addresses, so
        # this leaf would verify False (and nondeterministically) without
        # the length-prefixed recipe.
        result = verify_arrays(ragged, "4111")
        assert result["match"] is True
        assert set(result["computed"]) == {
            "5/morton",
            "5/h_li_tdigest",
            "5/h_li_tdigest_locations",
        }

    def test_vlen_recipe_is_the_frozen_one(self, ragged):
        # Hand-rolled, independent of hash_arrays: sha256 over
        # uint64_le(len) || payload per cell, in flat C order. Cell i holds i
        # float32s (cell 0 is empty, pinning the zero-length prefix).
        import hashlib

        import numpy as np

        digest = hashlib.sha256()
        for i in range(16):
            payload = np.arange(i, dtype="<f4").tobytes()
            digest.update(len(payload).to_bytes(stats.VLEN_LENGTH_PREFIX, "little"))
            digest.update(payload)
        hashes = hash_arrays(ragged, convention.leaf_path("4111"))
        assert hashes["5/h_li_tdigest"] == digest.hexdigest()

    def test_in_leaf_debris_is_out_of_o11_scope(self, ragged, tmp_path):
        """Non-zarr objects INSIDE the leaf (zagg's ``coverage.moc`` occupancy
        bitmap, a ``.zarr.status/`` prefix, editor/OS files) are not arrays, so
        they must neither raise nor move a hash. This is the class
        ``Group.members`` choked on — and choked on asymmetrically: zarr's own
        ``LocalStore`` warns and skips, obstore turns the same probe into a
        hard error, so the key walk is what makes the two agree."""
        rel = convention.leaf_path("4111")
        pristine = hash_arrays(ragged, rel)
        root = tmp_path / "store"
        shutil.copytree(ragged, root)
        leaf = root / rel
        (leaf / "coverage.moc").write_bytes(b"\x28\xb5\x2f\xfd")
        (leaf / ".zarr.status").mkdir()
        (leaf / ".zarr.status" / "last_write.json").write_text('{"run": 1}')
        (leaf / "5" / "notes.txt").write_text("scratch")
        assert hash_arrays(str(root), rel) == pristine

    def test_vlen_hash_is_stable_across_loads(self, ragged):
        # Two fresh opens decode into freshly allocated objects at different
        # addresses; a pointer-derived digest would differ here.
        rel = convention.leaf_path("4111")
        assert hash_arrays(ragged, rel) == hash_arrays(ragged, rel)

    def test_vlen_length_prefix_defeats_concatenation_collisions(self):
        # [b"ab", b"c"] and [b"a", b"bc"] share their concatenation; the
        # prefix is what keeps the digest injective.
        def recipe(payloads):
            import hashlib

            d = hashlib.sha256()
            for p in payloads:
                d.update(len(p).to_bytes(stats.VLEN_LENGTH_PREFIX, "little"))
                d.update(p)
            return d.hexdigest()

        assert recipe([b"ab", b"c"]) != recipe([b"a", b"bc"])

    def test_element_bytes_kinds(self):
        # bytes-likes pass through; str/ndarray cover the vlen-utf8 and typed
        # `vlen-array<T>` futures; an unhashable kind RAISES rather than
        # digesting a repr.
        import numpy as np

        assert stats._element_bytes(b"ab") == b"ab"
        assert stats._element_bytes(bytearray(b"ab")) == b"ab"
        assert stats._element_bytes(memoryview(b"ab")) == b"ab"
        assert stats._element_bytes(None) == b""
        assert stats._element_bytes("ab") == b"ab"
        assert (
            stats._element_bytes(np.arange(2, dtype=">u8")) == np.arange(2, dtype="<u8").tobytes()
        )
        with pytest.raises(ValueError, match="no O11 byte recipe"):
            stats._element_bytes(7)

    def test_hash_arrays_decoded_values(self, atl06):
        # Hashes are over decoded values (raw little-endian C-order bytes at
        # the declared dtype) — recomputable from the arrays themselves.
        import hashlib

        import numpy as np

        rel = convention.leaf_path("4111")
        hashes = hash_arrays(atl06, rel)
        count = np.arange(1, 17, dtype="<i8")
        assert hashes["5/count"] == hashlib.sha256(count.tobytes()).hexdigest()


class TestOverviewSidecarNames:
    """The zagg spec §5.3 rule: an overview sidecar is its basename's stem
    plus ``.stats.json``, at EVERY store spec revision."""

    def test_stem_grammar(self):
        assert overview_sidecar_key("all.zarr") == "all.stats.json"
        assert overview_sidecar_key("2019.zarr") == "2019.stats.json"
        assert overview_sidecar_path("4/3/3/1/2/2019.zarr") == "4/3/3/1/2/2019.stats.json"

    def test_spec_keyed_naming_would_collide(self):
        # WHY the rule is unconditional: one ancestor node holds every
        # window's overview, and the legacy grammar resolves them all — the
        # per-window ones included, since an overview basename carries no
        # `{id}_{window}` split — to one `stats.json` at that node.
        for basename in ("all.zarr", "2019.zarr", "2020.zarr"):
            assert stats_sidecar_key(basename, convention.HIVE_SPEC) == "stats.json"
        assert len({overview_sidecar_key(b) for b in ("all.zarr", "2019.zarr")}) == 2

    def test_malformed_basename_raises(self):
        # Inherits the /3 branch's strictness: no path escape, no `_`.
        for bad in ("all", "../all.zarr", "a_b.zarr"):
            with pytest.raises(ValueError):
                overview_sidecar_key(bad)


class TestReadOverviewStats:
    def test_reads_the_record_on_a_v1_store(self, ov_flat):
        # The case a spec-keyed name misses: `atl06` is `morton-hive/1`, so
        # `stats_sidecar_key` computes `stats.json` — while zagg wrote
        # `all.stats.json` beside the overview zarr.
        manifest = json.loads((OVERVIEW / "atl06" / "morton_hive.json").read_text())
        assert manifest["spec"] == convention.HIVE_SPEC
        assert not (OVERVIEW / "atl06" / "4" / "3" / "3" / "1" / "2" / "stats.json").exists()
        record = read_overview_stats(ov_flat, "43312")
        assert record["shard_key"] == convention.morton_word("43312")
        golden = _golden_stats("atl06", "6", "all")["43312"]
        assert golden["key"] == "all.stats.json"
        assert record["content_hashes"]["combined"] == golden["combined"]

    def test_addresses_any_declared_order(self, ov_flat):
        # Order 2 (cells at order 4): one ancestor node, three digits.
        record = read_overview_stats(ov_flat, "433")
        assert (
            record["content_hashes"]["combined"]
            == _golden_stats("atl06", "4", "all")["433"]["combined"]
        )

    def test_packed_word_node_accepted(self, ov_flat):
        # The `read_stats_rollup` addressing contract: word or decimal.
        assert read_overview_stats(ov_flat, convention.morton_word("43312")) == read_overview_stats(
            ov_flat, "43312"
        )

    def test_per_window_records(self, ov_windows):
        for window in ("2019", "2020"):
            record = read_overview_stats(ov_windows, "43312", window=window)
            assert record["window"] == window
            golden = _golden_stats("atl06_windows", "6", window)["43312"]
            assert record["content_hashes"]["combined"] == golden["combined"]
        # Distinct objects at ONE node — what the unconditional stem grammar
        # buys: the legacy name would have resolved both to `stats.json`.
        assert (
            read_overview_stats(ov_windows, "43312", window="2019")["content_hashes"]["combined"]
            != read_overview_stats(ov_windows, "43312", window="2020")["content_hashes"]["combined"]
        )

    def test_all_time_fold_is_addressable(self, ov_windows):
        # The reserved token IS spellable on the telemetry surface (contrast
        # espg/moczarr#30, which refuses it at the dataset openers): the
        # object exists, and one record cannot misreport coverage. `None` and
        # an explicit "all" name the same `all.zarr` sidecar.
        record = read_overview_stats(ov_windows, "43312")
        assert record == read_overview_stats(ov_windows, "43312", window="all")
        golden = _golden_stats("atl06_windows", "6", "all")["43312"]
        assert record["content_hashes"]["combined"] == golden["combined"]
        # It is a DIFFERENT object from either window's.
        assert record != read_overview_stats(ov_windows, "43312", window="2019")

    def test_unswept_node_reads_absent(self, ov_flat):
        # A legal ancestor decimal the sweep never wrote: no telemetry, and
        # (D9) that is never "no data".
        assert read_overview_stats(ov_flat, "43313") is None

    def test_deleted_sidecar_reads_absent(self, ov_flat, tmp_path):
        root = tmp_path / "store"
        shutil.copytree(ov_flat, root)
        (root / "4" / "3" / "3" / "1" / "2" / "all.stats.json").unlink()
        assert read_overview_stats(str(root), "43312") is None
        # The overview zarr itself is untouched — telemetry is not load-bearing.
        assert verify_overview_arrays(str(root), "43312")["computed"]

    @pytest.mark.parametrize("payload", ["{not json", '["a list"]'])
    def test_unusable_sidecar_reads_absent(self, ov_flat, tmp_path, payload):
        root = tmp_path / "store"
        shutil.copytree(ov_flat, root)
        (root / "4" / "3" / "3" / "1" / "2" / "all.stats.json").write_text(payload)
        assert read_overview_stats(str(root), "43312") is None

    def test_grouped_store_raises(self, pg3):
        # Name arithmetic is this module's loud surface: zagg's sweep writes
        # ancestor nodes one component per digit regardless of grouping, so
        # no key composed here would be the writer's.
        with pytest.raises(ValueError, match="path_grouping 3"):
            read_overview_stats(pg3, "4111")

    def test_non_hive_root_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a hive store root"):
            read_overview_stats(str(tmp_path), "433")


class TestVerifyOverviewArrays:
    def test_golden_match(self, ov_flat):
        # O11 parity with source leaves: zagg computes an overview's hashes
        # from the folded in-memory arrays with the same §5.1 recipe, so the
        # unchanged `hash_arrays` verifies them.
        result = verify_overview_arrays(ov_flat, "43312")
        assert result["match"] is True
        assert result["combined_match"] is True
        assert result["mismatched"] == []
        assert result["leaf"] == "4/3/3/1/2/all.zarr"
        assert set(result["computed"]) == {"6/morton", "6/count", "6/h_min", "6/h_max"}
        assert result["recorded"] == result["computed"]

    def test_every_committed_overview_verifies(self, ov_flat, ov_windows):
        for product, root in (("atl06", ov_flat), ("atl06_windows", ov_windows)):
            for order, per_key in OVERVIEW_GOLDEN["products"][product]["overviews"].items():
                for key, entry in per_key.items():
                    window = None if key == "all" else key
                    for node, recorded in entry["stats"].items():
                        result = verify_overview_arrays(root, node, window=window)
                        assert result["match"] is True, (product, order, key, node)
                        assert result["recorded_combined"] == recorded["combined"]

    def test_combined_hash_recomputes_to_the_golden(self, ov_flat):
        # Cross-implementation, not an echo: zagg recorded `combined`; this
        # recomputes it from the zarr bytes through moczarr's own recipe.
        result = verify_overview_arrays(ov_flat, "433")
        assert result["combined"] == combined_hash(hash_arrays(ov_flat, "4/3/3/all.zarr"))
        assert result["combined"] == _golden_stats("atl06", "4", "all")["433"]["combined"]

    def test_tamper_localizes_mismatch(self, ov_flat, tmp_path):
        root = tmp_path / "store"
        shutil.copytree(ov_flat, root)
        chunk = root / "4" / "3" / "3" / "1" / "2" / "all.zarr" / "6" / "count" / "c" / "0"
        data = bytearray(chunk.read_bytes())
        data[0] ^= 0xFF
        chunk.write_bytes(bytes(data))
        result = verify_overview_arrays(str(root), "43312")
        assert result["match"] is False
        assert result["mismatched"] == ["6/count"]
        assert result["combined"] != result["recorded_combined"]

    def test_nothing_recorded_is_none_not_a_pass(self, ov_flat, tmp_path):
        # The posture that matters most on a regenerable cache: an overview
        # whose sidecar records no hashes (an older sweep, or the fail-open
        # PUT) is UNVERIFIABLE, distinct from both verified and tampered.
        root = tmp_path / "store"
        shutil.copytree(ov_flat, root)
        sidecar = root / "4" / "3" / "3" / "1" / "2" / "all.stats.json"
        record = json.loads(sidecar.read_text())
        del record["content_hashes"]
        sidecar.write_text(json.dumps(record))
        result = verify_overview_arrays(str(root), "43312")
        assert result["match"] is None
        assert result["recorded"] is None and result["recorded_combined"] is None
        assert result["mismatched"] == []
        # Absence of the whole sidecar reads the same way — never a pass.
        sidecar.unlink()
        assert verify_overview_arrays(str(root), "43312")["match"] is None
        assert result["computed"], "an overview that EXISTS but records nothing"

    def test_absent_object_is_none_with_nothing_computed(self, ov_flat):
        # The other `match is None`: node 43313 was never swept, so the
        # composed path names no object and `hash_arrays` lists nothing.
        # Same verdict field, different fact — `computed` is what tells a
        # caller "there is no overview here" from "it exists, unverifiable".
        absent = verify_overview_arrays(ov_flat, "43313")
        assert absent["match"] is None and absent["computed"] == {}
        assert absent["combined"] == combined_hash({})
        present = verify_overview_arrays(ov_flat, "43312")
        assert present["computed"] and present["match"] is True

    def test_shares_the_source_leaf_recipe(self, ov_flat):
        # Same code path as `verify_arrays` below the addressing: the verdict
        # is reproducible from the public primitives.
        result = verify_overview_arrays(ov_flat, "43312")
        assert result["computed"] == hash_arrays(ov_flat, "4/3/3/1/2/all.zarr")
        assert result["combined"] == combined_hash(result["computed"])


class TestReadOverviewOrderStats:
    """The order-wide convenience, keyed like ``open_overview_order``."""

    @pytest.fixture()
    def ov_manifest(self):
        return json.loads((OVERVIEW / "atl06" / "morton_hive.json").read_text())

    @pytest.fixture()
    def ov_windows_manifest(self):
        return json.loads((OVERVIEW / "atl06_windows" / "morton_hive.json").read_text())

    def test_sweeps_every_materialized_node(self, ov_flat, ov_manifest):
        records = read_overview_order_stats(ov_flat, ov_manifest, 4)
        golden = _golden_stats("atl06", "6", "all")
        assert list(records) == ["43312", "43314", "43321", "43323"]
        for node, record in records.items():
            assert record["content_hashes"]["combined"] == golden[node]["combined"]
            assert record == read_overview_stats(ov_flat, node)

    def test_coarser_order_is_one_node(self, ov_flat, ov_manifest):
        records = read_overview_order_stats(ov_flat, ov_manifest, 2)
        assert list(records) == ["433"]

    def test_per_window_sweeps_differ(self, ov_windows, ov_windows_manifest):
        by_window = {
            window: read_overview_order_stats(ov_windows, ov_windows_manifest, 4, window=window)
            for window in (None, "2019", "2020")
        }
        for records in by_window.values():
            assert list(records) == ["43312", "43314", "43321", "43323"]
        combined = {
            window: {n: r["content_hashes"]["combined"] for n, r in records.items()}
            for window, records in by_window.items()
        }
        assert len({tuple(sorted(c.items())) for c in combined.values()}) == 3

    def test_omitted_window_is_the_fold_where_the_dataset_surface_raises(
        self, ov_windows, ov_windows_manifest, tmp_path
    ):
        # The ONE place this key means something different from
        # open_overview_order's, deliberately (espg/moczarr#30, PR #34 Q1):
        # the dataset surface refuses window=None on a windowed store, this
        # one addresses the all.zarr fold — read_overview_stats' meaning,
        # since a telemetry record has no tree node to misreport coverage
        # with. Pinned so the divergence stays a decision, not an accident.
        from moczarr.pyramid import open_overview_order

        assert read_overview_order_stats(
            ov_windows, ov_windows_manifest, 4
        ) == read_overview_order_stats(ov_windows, ov_windows_manifest, 4, window="all")
        with pytest.raises(ValueError, match="pass window="):
            open_overview_order(ov_windows, ov_windows_manifest, 4)
        # And its price, also pinned: where the folds do not exist (a
        # product declaring all_time: false, or one whose folds were deleted
        # — legal, §4.1 regenerable caches) the omitted window is an empty
        # mapping rather than the raise, so `{}` there is about the objects
        # addressed, not about the store's telemetry.
        root = tmp_path / "store"
        shutil.copytree(ov_windows, root)
        for path in root.rglob("all.zarr"):
            shutil.rmtree(path)
        for path in root.rglob("all.stats.json"):
            path.unlink()
        assert read_overview_order_stats(str(root), ov_windows_manifest, 4) == {}
        assert read_overview_order_stats(str(root), ov_windows_manifest, 4, window="2019")

    def test_keys_agree_with_open_overview_order(self, ov_flat, ov_manifest):
        # The two surfaces take the same key, so they must name the same
        # objects — the reason the candidate enumeration is shared.
        from moczarr.pyramid import node_objects, open_overview_order

        for order in ov_manifest["pyramid"]["overview"]["orders"]:
            ds = open_overview_order(ov_flat, ov_manifest, order)
            records = read_overview_order_stats(ov_flat, ov_manifest, order)
            assert list(records) == [entry["node"] for entry in node_objects(ds)]

    def test_missing_record_drops_the_node_not_the_object(self, ov_flat, ov_manifest, tmp_path):
        # A node whose sidecar is gone is ABSENT from the mapping — and the
        # overview object it belongs to is still there (telemetry is never
        # load-bearing, D9), which is why the two answers must be compared
        # rather than conflated.
        from moczarr.pyramid import node_objects, open_overview_order

        root = tmp_path / "store"
        shutil.copytree(ov_flat, root)
        (root / "4" / "3" / "3" / "1" / "2" / "all.stats.json").unlink()
        records = read_overview_order_stats(str(root), ov_manifest, 4)
        assert list(records) == ["43314", "43321", "43323"]
        ds = open_overview_order(str(root), ov_manifest, 4)
        assert [e["node"] for e in node_objects(ds)] == [
            "43312",
            "43314",
            "43321",
            "43323",
        ]

    def test_unswept_order_is_empty(self, ov_flat, ov_manifest):
        # Order 5 is a legal ancestor order the sweep never ran: the nodes
        # are nameable, none carries a record.
        assert read_overview_order_stats(ov_flat, ov_manifest, 5) == {}

    def test_no_root_moc_warns_and_empties(self, ov_flat, ov_manifest, tmp_path):
        # Silence here would be indistinguishable from "no telemetry at all".
        root = tmp_path / "store"
        shutil.copytree(ov_flat, root)
        (root / "coverage.moc").unlink()
        with pytest.warns(UserWarning, match="no usable root coverage.moc"):
            assert read_overview_order_stats(str(root), ov_manifest, 4) == {}

    def test_non_ancestor_order_raises(self, ov_flat, ov_manifest):
        with pytest.raises(ValueError, match="not an ancestor order"):
            read_overview_order_stats(ov_flat, ov_manifest, ov_manifest["shard_order"])

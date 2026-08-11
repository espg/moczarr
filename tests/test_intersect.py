"""Two-store occupancy intersection: golden fixtures, both API shapes.

The fixture stores are hand-built object-by-object (conftest's posture — no
zagg import, the wire format spelled out) with a KNOWN overlap, so every
expected cell set below is hand-computable from the leaf specs. The two API
shapes of zagg#422 open question 7 run behind one shared assertion
(``_both_shapes``): the flat compacted MOC expanded to the harmonized cell
order must equal the union of the per-leaf iterator's cells, cell-for-cell.
"""

import itertools
import json
import warnings

import numpy as np
import pytest
from numcodecs import Zstd

from moczarr import convention, intersect
from moczarr.convention import morton_word
from moczarr.intersect import iter_occupancy_and, occupancy_and

# Order-6 shards (southern base -5), all inside one order-4 branch.
S1 = "-5112333"  # bitmap in both stores
S2 = "-5112331"  # full in A, bitmap in B
S3 = "-5112334"  # bitmap in A, full in B
S4 = "-5112332"  # full in both (the compaction case)
S5 = "-5112343"  # A only
S6 = "-5112344"  # B only
S7 = "-5112342"  # shared shard, disjoint exact occupancy
S8 = "-5112341"  # degradation cases


def _words(*decimals):
    return np.sort(np.asarray([morton_word(d) for d in decimals], dtype=np.uint64))


def _expand(words, order):
    """Independent expansion: every member's descendants at ``order`` (string tails)."""
    out = []
    for w in np.asarray(words, dtype=np.uint64):
        dec = convention.morton_decimal(int(w))
        depth = order - convention.decimal_order(dec)
        assert depth >= 0, f"member {dec} is below order {order}"
        out.extend(morton_word(dec + "".join(t)) for t in itertools.product("1234", repeat=depth))
    return np.unique(np.asarray(out, dtype=np.uint64))


def _bitmap_payload(tails, depth):
    bits = np.zeros(4**depth, dtype=np.uint8)
    for tail in tails:
        assert len(tail) == depth
        bits[convention.decimal_rank("1" + tail)] = 1
    return bytes(Zstd(level=3).encode(np.packbits(bits).tobytes()))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def build_store(
    root,
    leaves,
    *,
    cell_order=8,
    shard_order=6,
    root_moc=True,
    window=None,
    envelope_cell_order=None,
):
    """A synthetic hive store from ``{shard: spec}`` leaf specs.

    Specs: ``("bitmap", [tails])`` exact sidecar; ``"full"`` whole-subtree
    envelope (no sidecar — D14); ``("box", [decimals])`` box-only envelope;
    ``("bitmap-nosidecar", [tails])`` a bitmap envelope whose sidecar object
    is missing; ``"stamp-only"`` a stamp without a coverage envelope;
    ``"debris"`` an unstamped leaf; ``"absent"`` root-MOC-listed with no
    object at all. ``window`` makes it a ``morton-hive/2`` store with every
    leaf at that window.

    ``envelope_cell_order`` skews the leaf ENVELOPES (and their bitmap
    depth) away from the manifest's ``cell_order`` — the wire shape a store
    rewritten at a new order leaves behind, and the one a reader must take
    from the envelope rather than the manifest.
    """
    spec = convention.HIVE_SPEC if window is None else convention.HIVE_SPEC_V2
    manifest = {
        "spec": spec,
        "dataset": {"short_name": "ATL03", "version": "007"},
        "cell_order": cell_order,
        "shard_order": shard_order,
        "split_schedule": [1] * shard_order,
        "pyramid": {"orders": [], "aggregation": {}},
        "generated_at": "2026-08-10T00:00:00+00:00",
    }
    if window is not None:
        manifest["temporal"] = {"schedule": "annual"}
    _write_json(root / convention.MANIFEST_NAME, manifest)
    if root_moc:
        ranked = sorted(leaves, key=convention.decimal_rank)
        _write_json(
            root / convention.ROOT_COVERAGE_NAME,
            {
                "spec": "morton-moc/1",
                "encoding": "ranges",
                "order": shard_order,
                "source": "dispatcher",
                "generated_at": "2026-08-10T00:00:00+00:00",
                "ranges": [[s, s] for s in ranked],
            },
        )
    env_order = cell_order if envelope_cell_order is None else envelope_cell_order
    depth = env_order - shard_order
    for shard, leaf_spec in leaves.items():
        leaf = root / convention.leaf_path(shard, window=window)
        if leaf_spec == "absent":
            continue
        if leaf_spec == "debris":
            _write_json(
                leaf / "zarr.json",
                {"zarr_format": 3, "node_type": "group", "attributes": {}},
            )
            continue
        coverage = {
            "spec": "morton-moc/1",
            "box": [shard, None, None, None],
            "cell_order": env_order,
            "source": "worker",
        }
        sidecar = None
        if leaf_spec == "full":
            coverage["encoding"] = "full"
        elif leaf_spec == "stamp-only":
            coverage = None
        elif leaf_spec[0] == "box":
            members = list(leaf_spec[1])
            coverage["box"] = members + [None] * (4 - len(members))
        else:  # "bitmap" / "bitmap-nosidecar"
            payload = _bitmap_payload(leaf_spec[1], depth)
            coverage.update(
                encoding="bitmap",
                sidecar=convention.COVERAGE_SIDECAR,
                nbytes=len(payload),
                raw_nbytes=-(-(4**depth) // 8),
            )
            if leaf_spec[0] == "bitmap":
                sidecar = payload
        stamp = {
            "spec": spec,
            "complete": True,
            "cells_with_data": 1,
            "granule_count": 1,
            "written_at": "2026-08-10T00:00:00+00:00",
        }
        if coverage is not None:
            stamp["coverage"] = coverage
        if window is not None:
            stamp["window"] = window
        _write_json(
            leaf / "zarr.json",
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {convention.COMMIT_ATTR: stamp},
            },
        )
        if sidecar is not None:
            (leaf / convention.COVERAGE_SIDECAR).write_bytes(sidecar)
    return str(root)


#: Store A (cell order 8): the finer-celled side of the golden pair.
LEAVES_A = {
    S1: ("bitmap", ["11", "12", "23", "44"]),
    S2: "full",
    S3: ("bitmap", ["11"]),
    S4: "full",
    S5: ("bitmap", ["11"]),
    S7: ("bitmap", ["11"]),
}
#: Store B (cell order 7): the coarser-celled side.
LEAVES_B = {
    S1: ("bitmap", ["1", "3", "4"]),
    S2: ("bitmap", ["2", "3"]),
    S3: "full",
    S4: "full",
    S6: ("bitmap", ["1"]),
    S7: ("bitmap", ["2"]),
}
#: Hand-computed golden intersection at the coarser order 7, per region:
#: A's S1 cells OR-coarsen 4:1 to {1, 2, 4}, AND B's {1, 3, 4} -> {1, 4};
#: full-in-A S2 keeps B's exact {2, 3}; full-in-B S3 keeps A's coarsened
#: {1}; full-in-both S4 is its whole depth-1 subtree; S7's exact sets are
#: disjoint (A's 11 -> 1, B's 2) and S5/S6 are unshared.
EXPECTED = {
    S2: [S2 + "2", S2 + "3"],
    S4: [S4 + d for d in "1234"],
    S1: [S1 + "1", S1 + "4"],
    S3: [S3 + "1"],
}


@pytest.fixture()
def golden_pair(tmp_path):
    root_a = build_store(tmp_path / "a", LEAVES_A, cell_order=8)
    root_b = build_store(tmp_path / "b", LEAVES_B, cell_order=7)
    return root_a, root_b


def _both_shapes(root_a, root_b, out_order, **kwargs):
    """Run BOTH API shapes and pin their cell-for-cell agreement."""
    got = dict(iter_occupancy_and(root_a, root_b, **kwargs))
    moc = occupancy_and(root_a, root_b, **kwargs)
    cells = np.unique(np.concatenate(list(got.values()))) if got else np.empty(0, dtype=np.uint64)
    np.testing.assert_array_equal(_expand(moc, out_order), cells)
    return got, moc


class TestGoldenPair:
    def test_mixed_order_intersection(self, golden_pair):
        got, moc = _both_shapes(*golden_pair, out_order=7)
        # Regions ascend in packed-word order (S2 < S4 < S1 < S3 by rank).
        assert list(got) == [morton_word(s) for s in (S2, S4, S1, S3)]
        for shard, cells in EXPECTED.items():
            np.testing.assert_array_equal(got[morton_word(shard)], _words(*cells))
        # The flat MOC compacts the full-in-both region to its shard word.
        np.testing.assert_array_equal(
            moc,
            _words(S4, *(c for s, cells in EXPECTED.items() if s != S4 for c in cells)),
        )

    def test_swapped_operands_agree(self, golden_pair):
        root_a, root_b = golden_pair
        np.testing.assert_array_equal(occupancy_and(root_a, root_b), occupancy_and(root_b, root_a))

    def test_cells_are_sorted_uint64(self, golden_pair):
        for _region, cells in iter_occupancy_and(*golden_pair):
            assert cells.dtype == np.uint64
            assert np.all(cells[1:] > cells[:-1])

    def test_full_leaves_never_pay_a_sidecar_get(self, golden_pair, monkeypatch):
        # The "full" short-circuit: no sidecar GET is even attempted for a
        # full-encoded leaf — only bitmap-encoded leaves reach the decoder.
        seen = []
        real = intersect.read_coverage_bitmap

        def counting(store_root, rel, **kwargs):
            seen.append((store_root, rel.rsplit("/", 1)[-1]))
            return real(store_root, rel, **kwargs)

        monkeypatch.setattr(intersect, "read_coverage_bitmap", counting)
        root_a, root_b = golden_pair
        _both_shapes(root_a, root_b, out_order=7)
        full = {(root_a, f"{S2}.zarr"), (root_a, f"{S4}.zarr")}
        full |= {(root_b, f"{S3}.zarr"), (root_b, f"{S4}.zarr")}
        assert full.isdisjoint(seen)
        assert (root_a, f"{S1}.zarr") in seen and (root_b, f"{S1}.zarr") in seen

    def test_aoi_prefilters_shards(self, golden_pair):
        got, _moc = _both_shapes(*golden_pair, out_order=7, aoi=[S1])
        assert list(got) == [morton_word(S1)]
        np.testing.assert_array_equal(got[morton_word(S1)], _words(*EXPECTED[S1]))

    def test_walk_fallback_without_root_moc(self, tmp_path):
        root_a = build_store(tmp_path / "a", LEAVES_A, cell_order=8, root_moc=False)
        root_b = build_store(tmp_path / "b", LEAVES_B, cell_order=7, root_moc=False)
        got, _moc = _both_shapes(root_a, root_b, out_order=7)
        assert set(got) == {morton_word(s) for s in EXPECTED}


class TestEqualOrders:
    def test_no_coarsening(self, tmp_path):
        root_a = build_store(tmp_path / "a", LEAVES_A, cell_order=8)
        root_c = build_store(tmp_path / "c", {S1: ("bitmap", ["11", "23"])}, cell_order=8)
        got, _moc = _both_shapes(root_a, root_c, out_order=8)
        assert list(got) == [morton_word(S1)]
        np.testing.assert_array_equal(got[morton_word(S1)], _words(S1 + "11", S1 + "23"))


class TestDifferentShardOrders:
    def test_regions_at_the_finer_shard_order(self, tmp_path):
        # A shards at order 6; D shards at order 7 under A's S1. Regions are
        # D's shards, each restricted against A's ONE cached leaf.
        root_a = build_store(tmp_path / "a", LEAVES_A, cell_order=8)
        root_d = build_store(
            tmp_path / "d",
            {S1 + "1": ("bitmap", ["2", "3"]), S1 + "3": ("bitmap", ["1"])},
            cell_order=8,
            shard_order=7,
        )
        got, _moc = _both_shapes(root_a, root_d, out_order=8)
        # Region S1+"1": A's cells there {11, 12} AND D's {12, 13} -> {12}.
        # Region S1+"3": A has nothing under it -> skipped.
        assert list(got) == [morton_word(S1 + "1")]
        np.testing.assert_array_equal(got[morton_word(S1 + "1")], _words(S1 + "12"))

    def _mixed_pair(self, tmp_path):
        """Both harmonizations at once — the issue's ATL03-o19-vs-GEDI-o18 shape.

        A shards at order 6 and cells at o9; D shards at order 7 and cells at
        o8. So regions sit at order 7 (D's leaves), results at order 8, and
        A's cells cross BOTH steps: restricted to a region by ancestor
        equality at o7, then 4:1 OR-coarsened o9 -> o8.
        """
        root_a = build_store(tmp_path / "a", {S1: ("bitmap", ["111", "123", "311"])}, cell_order=9)
        root_d = build_store(
            tmp_path / "d",
            {S1 + "1": ("bitmap", ["1", "3"]), S1 + "3": ("bitmap", ["2"])},
            cell_order=8,
            shard_order=7,
        )
        return root_a, root_d

    def test_mixed_shard_and_cell_orders(self, tmp_path):
        got, _moc = _both_shapes(*self._mixed_pair(tmp_path), out_order=8)
        # Region S1+"1": A's o9 cells under it are {111, 123}, which coarsen
        # to {11, 12}; D's leaf there holds {11, 13} -> {11}.
        # Region S1+"3": A's {311} coarsens to {31}, D's leaf holds {32} ->
        # empty, so the region is skipped.
        assert list(got) == [morton_word(S1 + "1")]
        np.testing.assert_array_equal(got[morton_word(S1 + "1")], _words(S1 + "11"))

    def test_swapped_operands_agree_on_the_mixed_pair(self, tmp_path):
        # _shared_regions is the one function whose two branches are
        # asymmetric by construction, so pin symmetry for BOTH shapes here
        # rather than only on the equal-shard-order golden pair.
        root_a, root_d = self._mixed_pair(tmp_path)
        forward = dict(iter_occupancy_and(root_a, root_d))
        reverse = dict(iter_occupancy_and(root_d, root_a))
        assert list(forward) == list(reverse)
        for region, cells in forward.items():
            np.testing.assert_array_equal(reverse[region], cells)
        np.testing.assert_array_equal(occupancy_and(root_a, root_d), occupancy_and(root_d, root_a))

    def test_coarse_full_leaf_spans_several_fine_regions(self, tmp_path):
        # ONE coarse "full" leaf covers BOTH of the fine store's shards: the
        # short-circuit answers each region separately off that one cached
        # leaf, and each keeps its own restricted intersection.
        root_a = build_store(tmp_path / "a", {S1: "full"}, cell_order=8)
        root_d = build_store(
            tmp_path / "d",
            {S1 + "1": ("bitmap", ["2"]), S1 + "3": ("bitmap", ["1"])},
            cell_order=8,
            shard_order=7,
        )
        got, _moc = _both_shapes(root_a, root_d, out_order=8)
        assert list(got) == [morton_word(S1 + "1"), morton_word(S1 + "3")]
        np.testing.assert_array_equal(got[morton_word(S1 + "1")], _words(S1 + "12"))
        np.testing.assert_array_equal(got[morton_word(S1 + "3")], _words(S1 + "31"))


class TestEnvelopeOrderAuthority:
    """The decoded cells' order is the leaf ENVELOPE's, never the manifest's."""

    def test_finer_envelope_than_manifest_still_coarsens(self, tmp_path):
        # K's manifest says cell_order 7 but its leaf envelope (and sidecar)
        # is o8 — the harmonized order is still 7, so K's o8 cell S1+"11"
        # must coarsen 4:1 to S1+"1" and meet L's genuine o7 cell there.
        # Gating the coarsening on the manifest leaves the cells at o8,
        # where no word can ever be bit-equal to an o7 word: a silent {}.
        root_k = build_store(
            tmp_path / "k", {S1: ("bitmap", ["11"])}, cell_order=7, envelope_cell_order=8
        )
        root_l = build_store(tmp_path / "l", {S1: ("bitmap", ["1"])}, cell_order=7)
        got, _moc = _both_shapes(root_k, root_l, out_order=7)
        assert list(got) == [morton_word(S1)]
        np.testing.assert_array_equal(got[morton_word(S1)], _words(S1 + "1"))

    def test_coarser_envelope_than_manifest_degrades(self, tmp_path):
        # The reverse skew: M's envelope is o7 while the harmonized order is
        # o8, so its cells name whole subtrees there — refining them would
        # invent occupancy. They ride the conservative cover path instead:
        # S1+"1" caps N's exact cells to S1+"11" (S1+"23" is outside).
        root_m = build_store(
            tmp_path / "m", {S1: ("bitmap", ["1"])}, cell_order=8, envelope_cell_order=7
        )
        root_n = build_store(tmp_path / "n", {S1: ("bitmap", ["11", "23"])}, cell_order=8)
        with pytest.warns(UserWarning, match="cell_order 7 is below"):
            got, _moc = _both_shapes(root_m, root_n, out_order=8)
        np.testing.assert_array_equal(got[morton_word(S1)], _words(S1 + "11"))


class TestDegradation:
    def _coarse(self, tmp_path):
        return build_store(tmp_path / "f", {S8: ("bitmap", ["1", "2"])}, cell_order=7)

    def test_box_only_leaf_is_conservative_and_warns(self, tmp_path):
        root_e = build_store(tmp_path / "e", {S8: ("box", [S8 + "1"])}, cell_order=8)
        with pytest.warns(UserWarning, match="conservative"):
            got, _moc = _both_shapes(root_e, self._coarse(tmp_path), out_order=7)
        # Box member S8+"1" caps the superset: only B's cell under it survives.
        np.testing.assert_array_equal(got[morton_word(S8)], _words(S8 + "1"))

    def test_stamp_without_envelope_is_conservative_full(self, tmp_path):
        root_g = build_store(tmp_path / "g", {S8: "stamp-only"}, cell_order=8)
        with pytest.warns(UserWarning, match="conservative"):
            got, _moc = _both_shapes(root_g, self._coarse(tmp_path), out_order=7)
        # Whole-shard stand-in: the exact side's cells pass through.
        np.testing.assert_array_equal(got[morton_word(S8)], _words(S8 + "1", S8 + "2"))

    def test_missing_sidecar_degrades_to_the_box(self, tmp_path):
        root_h = build_store(tmp_path / "h", {S8: ("bitmap-nosidecar", ["11"])}, cell_order=8)
        with pytest.warns(UserWarning, match="conservative"):
            got, _moc = _both_shapes(root_h, self._coarse(tmp_path), out_order=7)
        # The envelope's box is the whole shard -> the exact side passes through.
        np.testing.assert_array_equal(got[morton_word(S8)], _words(S8 + "1", S8 + "2"))

    def test_skip_drops_degraded_leaves_entirely(self, tmp_path):
        # degrade="skip": the box-only S8 leaf contributes NOTHING, so its
        # region disappears rather than answering a superset, and nothing
        # warns — everything yielded is exact (the discriminator a per-cell
        # consumer needs, which one call-scoped warning cannot give).
        root_e = build_store(
            tmp_path / "e", {S1: ("bitmap", ["11"]), S8: ("box", [S8 + "1"])}, cell_order=8
        )
        root_f = build_store(
            tmp_path / "f", {S1: ("bitmap", ["1"]), S8: ("bitmap", ["1", "2"])}, cell_order=7
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got, _moc = _both_shapes(root_e, root_f, out_order=7, degrade="skip")
        assert list(got) == [morton_word(S1)]
        np.testing.assert_array_equal(got[morton_word(S1)], _words(S1 + "1"))

    def test_raise_names_the_first_degraded_leaf(self, tmp_path):
        root_e = build_store(tmp_path / "e", {S8: ("box", [S8 + "1"])}, cell_order=8)
        coarse = self._coarse(tmp_path)
        with pytest.raises(ValueError, match=f"{S8}.zarr of "):
            occupancy_and(root_e, coarse, degrade="raise")
        with pytest.raises(ValueError, match="degrade='raise'"):
            list(iter_occupancy_and(root_e, coarse, degrade="raise"))

    def test_unknown_degrade_policy_raises_at_call_time(self, tmp_path, golden_pair):
        with pytest.raises(ValueError, match="degrade='nope'"):
            iter_occupancy_and(*golden_pair, degrade="nope")

    def test_degraded_against_exact_never_expands(self, tmp_path, monkeypatch):
        # A conservative cover stays a cover: the AND against an exact side
        # is a containment filter over the exact side's cells, so no subtree
        # is ever materialized. (Expanding the cover to every cell at the
        # harmonized order is 4^depth words per degraded leaf — GB-scale at
        # ATL03 depths, for an answer bounded by the other side.)
        calls = []
        real = intersect._expand_to
        monkeypatch.setattr(
            intersect, "_expand_to", lambda words, order: calls.append(order) or real(words, order)
        )
        root_e = build_store(tmp_path / "e", {S8: ("box", [S8 + "1"])}, cell_order=8)
        with pytest.warns(UserWarning, match="conservative"):
            got, _moc = _both_shapes(root_e, self._coarse(tmp_path), out_order=7)
        np.testing.assert_array_equal(got[morton_word(S8)], _words(S8 + "1"))
        assert calls == []

    def test_debris_and_absent_leaves_contribute_nothing(self, tmp_path):
        root_i = build_store(
            tmp_path / "i",
            {S1: ("bitmap", ["11"]), S2: "debris", S3: "absent"},
            cell_order=8,
        )
        root_j = build_store(
            tmp_path / "j",
            {S1: ("bitmap", ["1"]), S2: "full", S3: "full"},
            cell_order=7,
        )
        got, _moc = _both_shapes(root_i, root_j, out_order=7)
        assert list(got) == [morton_word(S1)]
        np.testing.assert_array_equal(got[morton_word(S1)], _words(S1 + "1"))


class TestWindows:
    def test_windowed_store_selects_the_window(self, tmp_path, golden_pair):
        root_a, root_b = golden_pair
        root_w = build_store(tmp_path / "w", LEAVES_B, cell_order=7, window="2019")
        got, _moc = _both_shapes(root_a, root_w, out_order=7, window_b="2019")
        baseline = dict(iter_occupancy_and(root_a, root_b))
        assert list(got) == list(baseline)
        for region, cells in baseline.items():
            np.testing.assert_array_equal(got[region], cells)

    def test_windowed_store_requires_a_window(self, tmp_path, golden_pair):
        root_a, _root_b = golden_pair
        root_w = build_store(tmp_path / "w", {S1: ("bitmap", ["1"])}, cell_order=7, window="2019")
        with pytest.raises(ValueError, match="window"):
            list(iter_occupancy_and(root_a, root_w))


class TestEdges:
    def test_empty_intersection(self, tmp_path):
        root_a = build_store(tmp_path / "a", {S5: ("bitmap", ["11"])}, cell_order=8)
        root_b = build_store(tmp_path / "b", {S6: ("bitmap", ["1"])}, cell_order=7)
        assert list(iter_occupancy_and(root_a, root_b)) == []
        moc = occupancy_and(root_a, root_b)
        assert moc.size == 0 and moc.dtype == np.uint64

    def test_incomposable_orders_raise(self, tmp_path):
        # Depth-0 store (cells AT shard order 6) against a store sharded at
        # order 7: no shared region has a cell representation in both.
        root_a = build_store(tmp_path / "a", {}, cell_order=6, shard_order=6)
        root_b = build_store(tmp_path / "b", {}, cell_order=9, shard_order=7)
        with pytest.raises(ValueError, match="do not compose"):
            occupancy_and(root_a, root_b)

    def test_missing_manifest_raises(self, tmp_path, golden_pair):
        root_a, _root_b = golden_pair
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="morton_hive.json"):
            occupancy_and(root_a, str(empty))

    def test_iterator_errors_fire_at_call_time(self, tmp_path, golden_pair):
        # iter_occupancy_and opens and validates both stores EAGERLY and
        # returns a generator over the prepared state — so the documented
        # ValueErrors land at the call, not at whatever frame first calls
        # next() (note: no list() below).
        root_a, _root_b = golden_pair
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="morton_hive.json"):
            iter_occupancy_and(root_a, str(empty))
        root_c = build_store(tmp_path / "c", {}, cell_order=6, shard_order=6)
        root_d = build_store(tmp_path / "d", {}, cell_order=9, shard_order=7)
        with pytest.raises(ValueError, match="do not compose"):
            iter_occupancy_and(root_c, root_d)

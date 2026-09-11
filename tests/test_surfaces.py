"""Percentile surfaces + the picker ladder: the gridlook feeder (issue #21).

Two acceptance laws under test. (1) Evaluation parity: every value in a
``(quantile, cells)`` surface IS ``zagg.stats.tdigest.quantile_from_tdigest``
on that cell's stored digest — per cell, per quantile, on the shared spec
fixtures — with the standing digest traps covered by hand-built bytes (the
empty cell, the single-centroid digest, the 0/1 endpoints, and the
unsorted-concatenation guard: centroids re-sorted by mean before the
interp-based kernel). (2) The ladder contract (gridlook#10 Phase 3):
declared orders, materialized orders (declared ≠ materialized is the
live-store failure mode), and per-order resolution, typed.

Fixture split mirrors ``test_level.py``: the vendored ``zagg-pyramid/2``
spec fixtures (``spec/temporal`` — the one with a root ``coverage.moc``;
``spec/kitchen_sink`` for the strata pair) and the zagg-swept ``/1``
``overview_hive`` stores. The live class at the bottom is env-gated
(``MOCZARR_LIVE_TESTS=1``) and bounded to one shard's column — it measures
the read+evaluate timing the issue's performance posture asks for.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moczarr import open_level, read_manifest
from moczarr.pyramid import OrderPresence
from moczarr.ragged import RAGGED_ATTR, RAGGED_SPEC, decode_cell, parse_ragged_attrs
from moczarr.surfaces import (
    DEFAULT_QUANTILES,
    SURFACE_ATTR,
    Ladder,
    LadderLevel,
    open_surface,
    quantile_surface,
    read_ladder,
)

zagg_tdigest = pytest.importorskip(
    "zagg.stats.tdigest", reason="surfaces evaluate through the moczarr[zagg] extra"
)

DATA = Path(__file__).parent / "data"
TEMPORAL = str(DATA / "spec" / "temporal")
KITCHEN = str(DATA / "spec" / "kitchen_sink")
MINIMAL = str(DATA / "spec" / "minimal")
OVERVIEW = str(DATA / "overview_hive" / "atl06")
WINDOWED = str(DATA / "overview_hive" / "atl06_windows")
SERC = str(DATA / "serc_hive")

#: The §1.2 attrs block a hand-built t-digest variable carries (spec grammar).
_DIGEST_ATTRS = {
    RAGGED_ATTR: {"spec": RAGGED_SPEC, "element": {"dtype": "float32", "shape": [-1, 2]}}
}


def _encode(digest) -> bytes:
    """One cell's ``(n, 2)`` centroids as zagg-ragged/1 vlen bytes."""
    return np.asarray(digest, dtype="<f4").tobytes()


def _level(cells, *, field="h_tdigest", extra=None):
    """A hand-built level-model Dataset: one vlen digest field on ``cells``.

    ``cells`` is a list of centroid lists; the two EMPTY spellings stay
    distinct in the object array — ``None`` is kept as the ``None`` object
    (``decode_cell``'s own branch) and ``[]`` encodes to ``b""`` — so a test
    naming both really covers both. ``extra`` maps more variable names to
    either digest cell lists (encoded the same way) or dense arrays (passed
    through).
    """
    n = len(cells)

    def _vlen(payloads):
        raw = np.empty(len(payloads), dtype=object)
        for i, d in enumerate(payloads):
            raw[i] = d if d is None else _encode(d)
        return xr.DataArray(raw, dims=("cells",), attrs=dict(_DIGEST_ATTRS))

    data = {field: _vlen(cells)}
    for name, value in (extra or {}).items():
        if isinstance(value, list):
            data[name] = _vlen(value)
        else:
            data[name] = xr.DataArray(np.asarray(value), dims=("cells",))
    morton = np.arange(100, 100 + n, dtype=np.uint64)
    return xr.Dataset(data, coords={"morton": ("cells", morton)})


class TestQuantileSurface:
    def test_values_are_the_zagg_kernel_per_cell(self):
        # The issue #21 acceptance: dense output matches quantile_from_tdigest
        # per cell on a shared fixture — here the temporal store's native
        # level, whose digests are real writer bytes.
        ds = open_level(TEMPORAL, 6)
        surf = quantile_surface(ds, "h_tdigest")
        element = parse_ragged_attrs(ds["h_tdigest"].attrs, field="h_tdigest")
        raw = ds["h_tdigest"].values
        assert surf["h_tdigest"].dims == ("quantile", "cells")
        assert surf.sizes["cells"] == ds.sizes["cells"] > 0
        populated = 0
        for j in range(surf.sizes["cells"]):
            digest = decode_cell(raw[j], element)
            for i, q in enumerate(DEFAULT_QUANTILES):
                got = float(surf["h_tdigest"].values[i, j])
                if len(digest) == 0:
                    assert np.isnan(got)
                else:
                    populated += 1
                    assert got == zagg_tdigest.quantile_from_tdigest(digest, q)
        assert populated > 0  # the fixture holds real digests
        # morton is carried verbatim, as plain values.
        np.testing.assert_array_equal(
            surf["morton"].values, np.asarray(ds["morton"].values, dtype=np.uint64)
        )

    def test_quantile_is_a_real_dimension_in_vanilla_xarray(self, tmp_path):
        # The output round-trips through plain zarr and reopens in vanilla
        # xarray with `quantile` as an indexable dimension coordinate —
        # nothing moczarr-specific survives materialization.
        surf = quantile_surface(_level([[[1.0, 1.0]], [[2.0, 1.0]]]), "h_tdigest")
        surf.to_zarr(tmp_path / "surf.zarr", mode="w")
        back = xr.open_zarr(tmp_path / "surf.zarr")
        assert list(back["quantile"].values) == list(DEFAULT_QUANTILES)
        sel = back["h_tdigest"].sel(quantile=0.5)
        np.testing.assert_array_equal(sel.values, [1.0, 2.0])
        # …and the metadata it wrote is STRICT JSON. The default fill is NaN,
        # which Python's json emits as a bare `NaN` literal (allow_nan) and
        # `JSON.parse` refuses — taking the array's shape/dtype/chunk grid
        # down with it in the viewer this module feeds.
        raw = (tmp_path / "surf.zarr" / "h_tdigest" / "zarr.json").read_text()

        def _bare(token):
            raise AssertionError(f"non-JSON literal {token!r} in the written zarr.json")

        meta = json.loads(raw, parse_constant=_bare)
        assert meta["attributes"][SURFACE_ATTR]["fill"] == "NaN"

    def test_empty_cell_reads_fill(self):
        # Both empty spellings, kept distinct in the object array: the None
        # OBJECT (decode_cell's dedicated branch — nothing between
        # _digest_columns and it normalizes a None) and the b"" zero-length
        # payload. Both read the caller's fill.
        level = _level([None, [], [[3.0, 2.0]]])
        raw = level["h_tdigest"].values
        assert raw[0] is None and raw[1] == b""
        surf = quantile_surface(level, "h_tdigest")
        assert np.isnan(surf["h_tdigest"].values[:, 0]).all()
        assert np.isnan(surf["h_tdigest"].values[:, 1]).all()
        assert (surf["h_tdigest"].values[:, 2] == 3.0).all()

    def test_caller_fill_replaces_nan(self):
        surf = quantile_surface(_level([None]), "h_tdigest", fill=-9999.0)
        assert (surf["h_tdigest"].values == -9999.0).all()

    def test_single_centroid_digest_is_constant(self):
        # One centroid answers every quantile with its mean — endpoints too.
        surf = quantile_surface(_level([[[7.5, 4.0]]]), "h_tdigest", [0.0, 0.5, 1.0])
        assert (surf["h_tdigest"].values == 7.5).all()

    def test_quantile_endpoints_pin_the_extremes(self):
        # q=0 and q=1 land on the first/last centroid means (the kernel's
        # rank-0 and rank-(n-1) semantics) — pinned here so an off-by-one in
        # the evaluation loop cannot hide behind interior quantiles.
        digest = [[1.0, 3.0], [5.0, 2.0], [9.0, 1.0]]
        surf = quantile_surface(_level([digest]), "h_tdigest", [0.0, 1.0])
        assert surf["h_tdigest"].values[0, 0] == 1.0
        assert surf["h_tdigest"].values[1, 0] == 9.0

    def test_unsorted_digest_is_resorted_before_evaluation(self):
        # The standing trap: the interp-based kernel walks centroids in mean
        # order, so an unsorted payload (a concatenated fold, or debris)
        # must be re-sorted first — the median of {1, 3, 5} (unit weights)
        # is 3 regardless of storage order.
        surf = quantile_surface(_level([[[5.0, 1.0], [1.0, 1.0], [3.0, 1.0]]]), "h_tdigest", [0.5])
        assert surf["h_tdigest"].values[0, 0] == 3.0
        sorted_surf = quantile_surface(
            _level([[[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]]]), "h_tdigest", [0.5]
        )
        np.testing.assert_array_equal(surf["h_tdigest"].values, sorted_surf["h_tdigest"].values)

    def test_strata_merge_is_the_sorted_concatenation(self):
        # The merged total (issue #21's 2-way arm): per cell, the lossless
        # t-digest union — concatenate the strata's centroid sets, sort by
        # mean, evaluate. Parity is against the kernel on that exact union,
        # and the merge is order-independent by construction.
        signal = [[[2.0, 3.0], [8.0, 1.0]], [[1.0, 1.0]], None]
        noise = [[[5.0, 2.0]], None, None]
        ds = _level(signal, field="h_tdigest_signal", extra={"h_tdigest_noise": noise})
        surf = quantile_surface(ds, ("h_tdigest_signal", "h_tdigest_noise"))
        assert "h_tdigest" in surf  # the common-prefix default name
        flipped = quantile_surface(ds, ("h_tdigest_noise", "h_tdigest_signal"), name="h_tdigest")
        np.testing.assert_array_equal(surf["h_tdigest"].values, flipped["h_tdigest"].values)
        for j, (s, m) in enumerate(zip(signal, noise)):
            parts = [np.asarray(p, dtype=np.float32) for p in (s, m) if p]
            if not parts:
                assert np.isnan(surf["h_tdigest"].values[:, j]).all()
                continue
            union = np.concatenate(parts, axis=0)
            union = union[np.lexsort((union[:, 1], union[:, 0]))]
            for i, q in enumerate(DEFAULT_QUANTILES):
                assert surf["h_tdigest"].values[i, j] == zagg_tdigest.quantile_from_tdigest(
                    union, q
                )

    def test_merged_total_is_order_independent_under_mean_ties(self):
        # The tie case the flipped-argument law actually turns on: two strata
        # SHARING a centroid mean. A stable sort by mean alone keeps the
        # argument order there, and the kernel's cumsum(weights) rank walk
        # reads the two orders differently (q=0.5 -> 4.111 vs 5.889 here), so
        # the union is canonicalized on (mean, weight) instead.
        a = [[[1.0, 1.0], [5.0, 10.0]]]
        b = [[[5.0, 1.0], [9.0, 1.0]]]
        ds = _level(a, field="h_tdigest_a", extra={"h_tdigest_b": b})
        ab = quantile_surface(ds, ("h_tdigest_a", "h_tdigest_b"))
        ba = quantile_surface(ds, ("h_tdigest_b", "h_tdigest_a"))
        np.testing.assert_array_equal(ab["h_tdigest"].values, ba["h_tdigest"].values)
        # …and the canonical order is the one the kernel is handed.
        union = np.asarray([[1.0, 1.0], [5.0, 1.0], [5.0, 10.0], [9.0, 1.0]], dtype=np.float32)
        for i, q in enumerate(DEFAULT_QUANTILES):
            assert ab["h_tdigest"].values[i, 0] == zagg_tdigest.quantile_from_tdigest(union, q)

    def test_per_stratum_vs_merged_parity(self):
        # kitchen_sink's real writer bytes: per-stratum surfaces are the
        # single-field calls, and the merged total brackets them — its
        # extremes are the strata extremes, and its total span covers each
        # stratum's (the union holds every centroid of both).
        ds = open_level(KITCHEN, 6)
        signal = quantile_surface(ds, "h_tdigest_signal", [0.0, 0.5, 1.0])
        noise = quantile_surface(ds, "h_tdigest_noise", [0.0, 0.5, 1.0])
        merged = quantile_surface(ds, ("h_tdigest_signal", "h_tdigest_noise"), [0.0, 0.5, 1.0])
        s, m, u = (
            signal["h_tdigest_signal"].values,
            noise["h_tdigest_noise"].values,
            merged["h_tdigest"].values,
        )
        both = ~np.isnan(s[0]) & ~np.isnan(m[0])
        assert both.any()  # the fixture holds cells with both strata
        np.testing.assert_array_equal(u[0, both], np.minimum(s[0, both], m[0, both]))
        np.testing.assert_array_equal(u[2, both], np.maximum(s[2, both], m[2, both]))
        only_s = ~np.isnan(s[0]) & np.isnan(m[0])
        np.testing.assert_array_equal(u[:, only_s], s[:, only_s])

    def test_count_rides_along_unevaluated(self):
        counts = np.array([4, 0, 7], dtype=np.int32)
        ds = _level([[[1.0, 4.0]], None, [[2.0, 7.0]]], extra={"count": counts})
        surf = quantile_surface(ds, "h_tdigest", carry=("count",))
        assert surf["count"].dims == ("cells",)
        np.testing.assert_array_equal(surf["count"].values, counts)
        assert surf["count"].dtype == np.int32

    def test_dense_only_arm_needs_no_digest(self):
        ds = _level([None], extra={"count": np.array([3], dtype=np.int32)})
        surf = quantile_surface(ds, None, carry=("count",))
        assert "quantile" not in surf.dims
        assert list(surf.data_vars) == ["count"]
        with pytest.raises(ValueError, match="carry= must name at least one"):
            quantile_surface(ds, None)

    def test_carry_takes_a_bare_string_and_a_one_shot_iterable(self):
        # carry= is normalized exactly like field=: a bare string is ONE
        # name (not its characters), and a generator must survive both the
        # dense-arm probe and the carry loop — consuming it once dropped
        # every carried field silently on the field=None arm.
        ds = _level([None], extra={"count": np.array([3], dtype=np.int32)})
        assert list(quantile_surface(ds, None, carry="count").data_vars) == ["count"]
        gen = (n for n in ["count"])
        assert list(quantile_surface(ds, None, carry=gen).data_vars) == ["count"]
        digest = _level([[[1.0, 1.0]]], extra={"count": np.array([1], dtype=np.int32)})
        assert set(quantile_surface(digest, "h_tdigest", carry="count").data_vars) == {
            "h_tdigest",
            "count",
        }

    def test_surface_attrs_record_the_evaluation(self):
        surf = quantile_surface(_level([[[1.0, 1.0]]]), "h_tdigest", [0.5], fill=-1.0)
        block = surf["h_tdigest"].attrs[SURFACE_ATTR]
        assert block == {"fields": ["h_tdigest"], "quantiles": [0.5], "fill": -1.0}
        # A non-finite fill is recorded by NAME, the house spelling (zarr's
        # own "fill_value": "NaN", the manifest's fill entries) — a bare
        # float would be an unparseable JSON literal on the way out.
        for fill, spelling in ((np.nan, "NaN"), (np.inf, "Infinity"), (-np.inf, "-Infinity")):
            block = quantile_surface(_level([[[1.0, 1.0]]]), "h_tdigest", [0.5], fill=fill)[
                "h_tdigest"
            ].attrs[SURFACE_ATTR]
            assert block["fill"] == spelling

    def test_refusals_are_pointed(self):
        counts = np.array([1], dtype=np.int32)
        ds = _level([[[1.0, 1.0]]], extra={"count": counts})
        with pytest.raises(KeyError, match="not a variable of this level"):
            quantile_surface(ds, "absent")
        with pytest.raises(ValueError, match="dense fields need no evaluation"):
            quantile_surface(ds, "count")
        with pytest.raises(ValueError, match="carry= copies"):
            quantile_surface(ds, "h_tdigest", carry=("h_tdigest",))
        with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
            quantile_surface(ds, "h_tdigest", [1.5])
        with pytest.raises(ValueError, match="non-empty"):
            quantile_surface(ds, "h_tdigest", [])
        with pytest.raises(ValueError, match="names no digest field"):
            quantile_surface(ds, ())

    def test_companion_sibling_is_not_a_digest(self):
        # A locations/times sibling is ragged but (n,) uint64 — evaluating it
        # as (mean, weight) centroids would be a silent wrong answer.
        ds = open_level(TEMPORAL, 6)
        with pytest.raises(ValueError, match="not the t-digest"):
            quantile_surface(ds, "h_tdigest_locations")


class TestOpenSurface:
    def test_column_level_end_to_end(self):
        # The one-call feeder on the /2 column tier: same values as opening
        # the level and materializing in two steps.
        two_step = quantile_surface(open_level(TEMPORAL, 5), "h_tdigest", carry=("count",))
        one_call = open_surface(TEMPORAL, 5, "h_tdigest", carry=("count",))
        xr.testing.assert_identical(one_call, two_step)
        assert one_call["h_tdigest"].dims == ("quantile", "cells")
        assert one_call.attrs["zagg_level"]["artifact"] == "column"

    def test_unmaterialized_level_is_none_with_the_arms_warning(self):
        # Levels 4..1 are declared, unswept (the fixtures are deliberately
        # unmaterialized at ancestor nodes) — open_level's degrade posture
        # passes through: None, never an empty surface.
        with pytest.warns(UserWarning, match="no stamped"):
            assert open_surface(TEMPORAL, 4, "h_tdigest") is None

    def test_disjoint_aoi_is_the_schema_correct_empty_surface(self):
        # AOI scoping is the level model's: rows only, never the schema. A
        # disjoint AOI yields (len(quantiles), 0) with the coords intact.
        from moczarr.convention import morton_word

        away = np.array([morton_word("21213")], dtype=np.uint64)  # a disjoint shard subtree
        with pytest.warns(UserWarning):
            surf = open_surface(TEMPORAL, 6, "h_tdigest", aoi=away)
        assert surf is not None
        assert surf["h_tdigest"].shape == (len(DEFAULT_QUANTILES), 0)
        assert surf["morton"].size == 0

    def test_windowed_store_follows_the_d23_seam(self):
        # window= reaches the arm untouched: the windowed store serves its
        # per-window artifacts, and the reserved token is refused.
        surf = open_surface(WINDOWED, 6, None, window="2019", carry=("count", "h_max"))
        assert set(surf.data_vars) == {"count", "h_max"}
        assert surf.attrs["morton_hive"]["spec"] == "morton-hive/2"
        with pytest.raises(ValueError, match="reserved all-time token"):
            open_surface(WINDOWED, 6, None, window="all", carry=("count",))

    def test_all_time_reads_the_cross_window_folds(self):
        surf = open_surface(WINDOWED, 6, None, all_time=True, carry=("count",))
        assert surf.attrs["zagg_objects"][0]["window"] == "all"

    def test_level_model_knobs_thread_through(self):
        # open_level's WHOLE signature reaches the arm — including the three
        # that are not surface arguments: fabricate_cell_ids decides whether
        # the surface has a cell_ids coordinate at all, index_kind selects
        # the level's index, and xr_kwargs reaches xr.open_zarr (the read,
        # which is the wall at viewer scales).
        assert "cell_ids" in open_surface(TEMPORAL, 6, "h_tdigest").coords
        bare = open_surface(TEMPORAL, 6, "h_tdigest", fabricate_cell_ids=False)
        assert "cell_ids" not in bare.coords
        pandas_index = open_surface(
            TEMPORAL, 6, "h_tdigest", index_kind="pandas", fabricate_cell_ids=False
        )
        assert pandas_index["h_tdigest"].shape == bare["h_tdigest"].shape
        with pytest.raises(ValueError, match="index_kind='bogus'"):  # the level model's own refusal
            open_surface(TEMPORAL, 6, "h_tdigest", index_kind="bogus", fabricate_cell_ids=False)
        with pytest.raises(TypeError, match="open_zarr"):  # unfiltered, all the way down
            open_surface(TEMPORAL, 6, "h_tdigest", xr_kwargs={"not_an_open_zarr_kwarg": 1})

    def test_strata_store_native_level(self):
        # kitchen_sink through the store-addressed arm: the merged total at
        # the native order, count riding along.
        surf = open_surface(KITCHEN, 6, ("h_tdigest_signal", "h_tdigest_noise"), carry=("count",))
        assert surf["h_tdigest"].dims == ("quantile", "cells")
        populated = ~np.isnan(surf["h_tdigest"].values[0])
        np.testing.assert_array_equal(populated, surf["count"].values > 0)


class TestReadLadder:
    def test_v2_ladder_declared_vs_materialized(self):
        # The gridlook#10 failure mode, pinned: the /2 fixture declares the
        # dense ladder down to order 1, materializes the source + the one
        # column group — declared and materialized MUST disagree.
        ladder = read_ladder(TEMPORAL)
        assert ladder.declared == (6, 5, 4, 3, 2, 1)
        assert ladder.materialized == (6, 5)
        by_order = {lvl.cell_order: lvl for lvl in ladder.levels}
        assert by_order[6].artifact == "source" and by_order[6].materialized is True
        assert by_order[5].presence == OrderPresence(nodes=1, stamped=1)
        assert by_order[4].materialized is False
        assert by_order[4].presence == OrderPresence(nodes=1, stamped=0)

    def test_v1_ladder_is_fully_materialized(self):
        ladder = read_ladder(OVERVIEW)
        assert ladder.declared == (8, 6, 4)
        assert ladder.materialized == (8, 6, 4)
        by_order = {lvl.cell_order: lvl for lvl in ladder.levels}
        assert by_order[6].presence == OrderPresence(nodes=4, stamped=4)
        assert by_order[8].artifact == "source"

    def test_per_order_resolution_is_mortie(self):
        # Derived, never hardcoded: each entry carries order2res at its CELL
        # order (the resolution a reader gets, not the node order).
        from mortie import order2res

        for lvl in read_ladder(OVERVIEW).levels:
            assert lvl.resolution_km == float(order2res(lvl.cell_order))
        levels = read_ladder(OVERVIEW).levels
        assert levels[0].resolution_km < levels[-1].resolution_km  # finest first

    def test_windowed_probe_needs_a_window(self):
        with pytest.raises(ValueError, match="pass window=.*or probe=False"):
            read_ladder(WINDOWED)
        ladder = read_ladder(WINDOWED, window="2019")
        assert ladder.declared == (8, 6)
        assert ladder.materialized == (8, 6)

    def test_probe_false_is_declaration_only(self):
        # One manifest GET: every non-source level reads unknown (None), and
        # the materialized projection lists nothing it cannot vouch for.
        ladder = read_ladder(WINDOWED, probe=False)
        assert ladder.declared == (8, 6)
        assert ladder.materialized == (8,)
        assert {lvl.materialized for lvl in ladder.levels} == {True, None}

    def test_window_seams_are_unconditional(self):
        with pytest.raises(ValueError, match="unwindowed"):
            read_ladder(OVERVIEW, window="2019")
        with pytest.raises(ValueError, match="reserved all-time token"):
            read_ladder(WINDOWED, window="all", probe=False)

    def test_unprobeable_coverage_reads_unknown(self):
        # minimal has no root coverage.moc: candidates cannot be named, so
        # every probed level degrades to None — with the arms' warnings —
        # and only the source order stays materialized.
        with pytest.warns(UserWarning, match="coverage.moc"):
            ladder = read_ladder(MINIMAL)
        assert ladder.declared == (6, 5, 4, 3, 2, 1)
        assert ladder.materialized == (6,)
        assert {lvl.materialized for lvl in ladder.levels if lvl.artifact != "source"} == {None}

    def test_declared_off_store_is_the_one_level_ladder(self):
        ladder = read_ladder(SERC)
        assert len(ladder.levels) == 1
        (only,) = ladder.levels
        assert (only.cell_order, only.artifact, only.materialized) == (8, "source", True)

    def test_threading_manifest_and_typed_record(self):
        manifest = read_manifest(OVERVIEW)
        ladder = read_ladder(OVERVIEW, manifest=manifest)
        assert isinstance(ladder, Ladder) and isinstance(ladder.levels[0], LadderLevel)
        with pytest.raises(AttributeError):
            ladder.levels = ()  # frozen — the PyramidInfo posture


@pytest.mark.skipif(
    os.environ.get("MOCZARR_LIVE_TESTS", "").lower() not in {"1", "true", "yes"},
    reason="live S3 acceptance (anonymous, one shard's column group); set MOCZARR_LIVE_TESTS=1",
)
class TestLiveSurface:
    """The performance posture against the published ATL03 store.

    Bounded to ONE shard's §4.6 column (the audited four-field shard
    ``test_column.py`` pins): reads one column group and materializes the
    default quantile set for both strata merged — the exact per-order
    choropleth call gridlook's hive.py makes. Prints read/evaluate wall
    times for the PR record; asserts only sanity bounds so the test stays a
    check, not a benchmark gate.
    """

    ROOT = "s3://us-west-2.opendata.source.coop/englacial/zagg/demo/atl03_tdigest_o9.zarr"
    S3 = {"region": "us-west-2", "anonymous": True}
    FOUR_FIELD = "3133332144"  # audited 2026-09-10 (see test_column.py)

    def test_one_shard_column_surface_timing(self):
        from moczarr import open_column_order
        from moczarr.convention import morton_word

        manifest = read_manifest(self.ROOT, **self.S3)
        aoi = np.asarray([morton_word(self.FOUR_FIELD)], dtype=np.uint64)
        t0 = time.perf_counter()
        ds = open_column_order(self.ROOT, manifest, 13, aoi=aoi, **self.S3)
        _ = ds["h_tdigest_signal"].load()  # pull the vlen payloads inside the read timing
        t_read = time.perf_counter() - t0
        t0 = time.perf_counter()
        surf = quantile_surface(ds, ("h_tdigest_signal", "h_tdigest_noise"), carry=("count",))
        t_eval = time.perf_counter() - t0
        n = surf.sizes["cells"]
        populated = int((~np.isnan(surf["h_tdigest"].values[0])).sum())
        print(
            f"\nlive surface: {n} cells at group 13 (one o9 shard), {populated} populated; "
            f"read {t_read:.2f}s, evaluate {t_eval:.2f}s"
        )
        assert surf["h_tdigest"].shape == (len(DEFAULT_QUANTILES), n)
        assert 0 < populated <= n

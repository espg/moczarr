"""Percentile surfaces: the gridlook feeder (issue #21).

The evaluation acceptance law: every value in a
``(quantile, cells)`` surface IS ``zagg.stats.tdigest.quantile_from_tdigest``
on that cell's stored digest — per cell, per quantile, on the shared spec
fixtures — with the standing digest traps covered by hand-built bytes (the
empty cell, the single-centroid digest, the 0/1 endpoints, and the
unsorted-concatenation guard: centroids re-sorted by mean before the
interp-based kernel).

Fixture split mirrors ``test_level.py``: the vendored ``zagg-pyramid/2``
spec fixtures (``spec/temporal`` — the one with a root ``coverage.moc``;
``spec/kitchen_sink`` for the strata pair) and the zagg-swept ``/1``
``overview_hive`` stores.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from moczarr import open_level
from moczarr.ragged import RAGGED_ATTR, RAGGED_SPEC, decode_cell, parse_ragged_attrs
from moczarr.surfaces import (
    DEFAULT_QUANTILES,
    SURFACE_ATTR,
    open_surface,
    quantile_surface,
)

zagg_tdigest = pytest.importorskip(
    "zagg.stats.tdigest", reason="surfaces evaluate through the moczarr[zagg] extra"
)

DATA = Path(__file__).parent / "data"
TEMPORAL = str(DATA / "spec" / "temporal")
KITCHEN = str(DATA / "spec" / "kitchen_sink")
WINDOWED = str(DATA / "overview_hive" / "atl06_windows")

#: The §1.2 attrs block a hand-built t-digest variable carries (spec grammar).
_DIGEST_ATTRS = {
    RAGGED_ATTR: {"spec": RAGGED_SPEC, "element": {"dtype": "float32", "shape": [-1, 2]}}
}


def _encode(digest) -> bytes:
    """One cell's ``(n, 2)`` centroids as zagg-ragged/1 vlen bytes."""
    return np.asarray(digest, dtype="<f4").tobytes()


def _level(cells, *, field="h_tdigest", extra=None):
    """A hand-built level-model Dataset: one vlen digest field on ``cells``.

    ``cells`` is a list of centroid lists (``None`` = the absent/empty vlen
    cell); ``extra`` maps more variable names to either digest cell lists
    (encoded the same way) or dense arrays (passed through).
    """
    n = len(cells)

    def _vlen(payloads):
        raw = np.empty(len(payloads), dtype=object)
        for i, d in enumerate(payloads):
            raw[i] = b"" if d is None else _encode(d)
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

    def test_empty_cell_reads_fill(self):
        # Both empty spellings (b"" via None here, and a zero-length list).
        surf = quantile_surface(_level([None, [], [[3.0, 2.0]]]), "h_tdigest")
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
            union = union[np.argsort(union[:, 0], kind="stable")]
            for i, q in enumerate(DEFAULT_QUANTILES):
                assert surf["h_tdigest"].values[i, j] == zagg_tdigest.quantile_from_tdigest(
                    union, q
                )

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

    def test_surface_attrs_record_the_evaluation(self):
        surf = quantile_surface(_level([[[1.0, 1.0]]]), "h_tdigest", [0.5], fill=-1.0)
        block = surf["h_tdigest"].attrs[SURFACE_ATTR]
        assert block == {"fields": ["h_tdigest"], "quantiles": [0.5], "fill": -1.0}

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

    def test_strata_store_native_level(self):
        # kitchen_sink through the store-addressed arm: the merged total at
        # the native order, count riding along.
        surf = open_surface(KITCHEN, 6, ("h_tdigest_signal", "h_tdigest_noise"), carry=("count",))
        assert surf["h_tdigest"].dims == ("quantile", "cells")
        populated = ~np.isnan(surf["h_tdigest"].values[0])
        np.testing.assert_array_equal(populated, surf["count"].values > 0)

"""Native t-digest kernels + the zagg parity gate (issue #64).

Two layers. The native layer runs everywhere (pure numpy — the whole point
of issue #64 is that no reader function needs zagg): the algebra's own laws
(empty/single/endpoint semantics, the midpoint-of-weight-interval CDF
positions, k-way order independence, the delta cap). The parity layer runs
only under the demoted ``moczarr[zagg]`` extra and pins EXACT value equality
— ``==`` / ``assert_array_equal``, never ``np.isclose`` — between the native
kernels and ``zagg.stats.tdigest`` on: every digest stored in the in-tree
fixtures (harvested by their §1.2 attrs, not by a hardcoded list), k-way
merges of those digests under several deltas and permutations, and a seeded
random fuzz (duplicate means, tied ``(mean, weight)`` rows, tiny/huge
weights).
"""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from zarr.storage import LocalStore

from moczarr.ragged import read_ragged
from moczarr.tdigest import (
    DEFAULT_DELTA,
    cdf_from_tdigest,
    merge_tdigests_kway,
    quantile_from_tdigest,
)

DATA = Path(__file__).parent / "data"

HAS_ZAGG = importlib.util.find_spec("zagg") is not None
needs_zagg = pytest.mark.skipif(not HAS_ZAGG, reason="parity gate needs the moczarr[zagg] extra")

#: Quantile probes: both endpoints plus an interior sweep.
QS = [0.0, 0.02, 0.15, 0.25, 0.5, 0.75, 0.85, 0.98, 1.0]


def _digest(rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def _fixture_digest_fields() -> list[tuple[str, str]]:
    """Every ``(leaf_root, field_path)`` holding spec-§2.1 digest payloads.

    Discovered from the stores' own §1.2 attrs (``element == {"dtype":
    "float32", "shape": [-1, 2]}``), so a new fixture joins the parity gate
    by existing — no list to keep.
    """
    found = []
    for meta in sorted(DATA.rglob("zarr.json")):
        try:
            attrs = json.loads(meta.read_text()).get("attributes", {})
        except (OSError, json.JSONDecodeError):
            continue
        element = attrs.get("ragged", {}).get("element")
        if element == {"dtype": "float32", "shape": [-1, 2]}:
            field_dir = meta.parent
            root = field_dir.parent.parent
            found.append((str(root), f"{field_dir.parent.name}/{field_dir.name}"))
    return found


def _fixture_digests() -> list[np.ndarray]:
    """Every populated digest stored in the in-tree fixtures."""
    return [d for group in _fixture_digest_groups().values() for d in group]


def _fixture_digest_groups() -> dict[str, list[np.ndarray]]:
    """Fixture digests grouped per store root — the merge-legal unit.

    Spec §2.0 makes merges legal only between payloads sharing a ``weights``
    declaration (counts with counts, flux with flux), so the parity gate
    merges within one store's fields (the reader's real fold: strata of one
    leaf), never across stores — the flux fixture must not fold into the
    counts stores' digests.
    """
    groups: dict[str, list[np.ndarray]] = {}
    for root, field in _fixture_digest_fields():
        for _word, values in read_ragged(LocalStore(root), field):
            groups.setdefault(root, []).append(np.asarray(values, dtype=np.float32))
    return groups


class TestQuantileNative:
    def test_empty_digest_is_nan(self):
        assert np.isnan(quantile_from_tdigest(np.empty((0, 2), dtype=np.float32), 0.5))

    def test_single_centroid_answers_every_quantile(self):
        d = _digest([[7.5, 4.0]])
        for q in QS:
            assert quantile_from_tdigest(d, q) == 7.5

    def test_endpoints_are_the_extreme_means(self):
        d = _digest([[1.0, 3.0], [5.0, 2.0], [9.0, 1.0]])
        assert quantile_from_tdigest(d, 0.0) == 1.0
        assert quantile_from_tdigest(d, 1.0) == 9.0

    def test_unit_weights_recover_the_median(self):
        # {1, 3, 5} with unit weights: rank 1 falls in the middle centroid's
        # single-observation interval, which answers its mean exactly.
        d = _digest([[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]])
        assert quantile_from_tdigest(d, 0.5) == 3.0

    def test_interior_interpolates_between_mean_midpoints(self):
        # Two centroids, weights (2, 2), total 4: q = 1/3 targets rank 1.0,
        # the top of centroid 0's rank interval [0, 1] (frac = 1), whose
        # upper value is the midpoint of the adjacent means — (0 + 10)/2.
        d = _digest([[0.0, 2.0], [10.0, 2.0]])
        assert quantile_from_tdigest(d, 1.0 / 3.0) == 5.0

    def test_monotonic_in_q(self):
        rng = np.random.default_rng(7)
        means = np.sort(rng.normal(size=40))
        weights = rng.integers(1, 9, size=40).astype(np.float64)
        d = _digest(np.column_stack([means, weights]))
        got = [quantile_from_tdigest(d, q) for q in np.linspace(0.0, 1.0, 33)]
        assert got == sorted(got)


class TestCdfNative:
    def test_empty_digest_is_nan_shaped_like_x(self):
        empty = np.empty((0, 2), dtype=np.float32)
        assert np.isnan(cdf_from_tdigest(empty, 3.0))
        out = cdf_from_tdigest(empty, np.asarray([1.0, 2.0]))
        assert out.shape == (2,) and np.isnan(out).all()

    def test_single_centroid_is_a_step(self):
        d = _digest([[5.0, 3.0]])
        assert cdf_from_tdigest(d, 4.999) == 0.0
        assert cdf_from_tdigest(d, 5.0) == 3.0
        assert cdf_from_tdigest(d, 6.0) == 3.0

    def test_positions_are_weight_interval_midpoints_not_cumsum(self):
        # The standing trap, pinned: at the FIRST mean the cumulative weight
        # is w0/2 (the centroid's midpoint), not w0 (the raw cumsum).
        d = _digest([[0.0, 2.0], [1.0, 2.0]])
        assert cdf_from_tdigest(d, 0.0) == 1.0  # w0/2, not 2.0
        assert cdf_from_tdigest(d, 1.0) == 3.0  # w0 + w1/2

    def test_clamps_to_zero_and_total(self):
        d = _digest([[0.0, 2.0], [1.0, 3.0]])
        assert cdf_from_tdigest(d, -1.0) == 0.0
        assert cdf_from_tdigest(d, 2.0) == 5.0

    def test_scalar_in_scalar_out_array_in_array_out(self):
        d = _digest([[0.0, 1.0], [1.0, 1.0]])
        assert isinstance(cdf_from_tdigest(d, 0.5), float)
        out = cdf_from_tdigest(d, np.asarray([0.0, 0.5, 1.0]))
        assert isinstance(out, np.ndarray) and out.dtype == np.float64

    def test_monotonic_over_a_fixture_digest(self):
        digests = _fixture_digests()
        assert digests  # the tree carries real writer bytes
        d = max(digests, key=len)
        xs = np.linspace(float(d[:, 0].min()) - 1.0, float(d[:, 0].max()) + 1.0, 101)
        out = cdf_from_tdigest(d, xs)
        assert (np.diff(out) >= 0.0).all()
        assert out[0] == 0.0 and out[-1] == float(np.sum(d[:, 1], dtype=np.float64))


class TestKwayNative:
    def _random_digests(self, rng, k):
        out = []
        for _ in range(k):
            n = int(rng.integers(1, 60))
            means = np.sort(rng.normal(scale=10.0, size=n))
            weights = rng.integers(1, 50, size=n).astype(np.float64)
            out.append(_digest(np.column_stack([means, weights])))
        return out

    def test_all_empty_is_the_empty_digest(self):
        out = merge_tdigests_kway([np.empty((0, 2), dtype=np.float32)] * 3)
        assert out.shape == (0, 2) and out.dtype == np.float32

    def test_single_contributor_passes_through_uncompressed(self):
        d = _digest([[1.0, 2.0], [3.0, 4.0]])
        out = merge_tdigests_kway([np.empty((0, 2), dtype=np.float32), d])
        np.testing.assert_array_equal(out, d)
        assert out is not d  # a copy, never an alias

    def test_permutation_independence(self):
        rng = np.random.default_rng(11)
        digests = self._random_digests(rng, 6)
        base = merge_tdigests_kway(digests)
        for seed in range(4):
            perm = list(np.random.default_rng(seed).permutation(len(digests)))
            np.testing.assert_array_equal(merge_tdigests_kway([digests[i] for i in perm]), base)

    def test_permutation_independence_under_mean_ties(self):
        # The tie the (mean, weight) lexsort key exists for: two inputs
        # sharing a centroid mean with different weights.
        a = _digest([[1.0, 1.0], [5.0, 10.0]])
        b = _digest([[5.0, 1.0], [9.0, 1.0]])
        np.testing.assert_array_equal(merge_tdigests_kway([a, b]), merge_tdigests_kway([b, a]))

    def test_weight_conserved_means_sorted_delta_capped(self):
        rng = np.random.default_rng(13)
        digests = self._random_digests(rng, 40)
        total = sum(float(np.sum(d[:, 1], dtype=np.float64)) for d in digests)
        for delta in (8, 64, DEFAULT_DELTA):
            out = merge_tdigests_kway(digests, delta=delta)
            assert out.dtype == np.float32 and out.ndim == 2 and out.shape[1] == 2
            assert (np.diff(out[:, 0]) >= 0.0).all()
            np.testing.assert_allclose(np.sum(out[:, 1], dtype=np.float64), total, rtol=1e-6)
            # The k1 budget: one flat k-way pass holds the centroid count
            # near delta — most centroids span a full k-unit of k ∈
            # [0, delta], with steep-tail forced singletons pushing a few
            # past it (never the 2x a pairwise fold drifts to).
            assert len(out) <= 2 * delta


@needs_zagg
class TestZaggParity:
    """EXACT value equality against ``zagg.stats.tdigest`` (issue #64's gate)."""

    def _assert_eval_parity(self, digests):
        from zagg.stats import tdigest as zt

        assert digests
        for d in digests:
            for q in QS:
                native, ref = quantile_from_tdigest(d, q), zt.quantile_from_tdigest(d, q)
                assert native == ref or (np.isnan(native) and np.isnan(ref))
            if len(d):
                lo, hi = float(d[:, 0].min()), float(d[:, 0].max())
                xs = np.linspace(lo - 1.0, hi + 1.0, 41)
            else:
                xs = np.linspace(-1.0, 1.0, 5)
            np.testing.assert_array_equal(cdf_from_tdigest(d, xs), zt.cdf_from_tdigest(d, xs))
            x0 = float(xs[len(xs) // 2])
            native0, ref0 = cdf_from_tdigest(d, x0), zt.cdf_from_tdigest(d, x0)
            assert native0 == ref0 or (np.isnan(native0) and np.isnan(ref0))

    def test_every_fixture_digest_evaluates_identically(self):
        self._assert_eval_parity(_fixture_digests())

    def test_fixture_kway_merges_are_byte_identical(self):
        from zagg.stats import tdigest as zt

        groups = [g for g in _fixture_digest_groups().values() if len(g) > 1]
        assert groups  # the tree carries multi-digest stores
        rng = np.random.default_rng(17)
        for digests in groups:
            for delta in (8, 64, DEFAULT_DELTA):
                for _trial in range(3):
                    perm = [digests[i] for i in rng.permutation(len(digests))]
                    np.testing.assert_array_equal(
                        merge_tdigests_kway(perm, delta=delta),
                        zt.merge_tdigests_kway(perm, delta=delta),
                    )

    def test_merged_fixture_strata_evaluate_identically(self):
        # The reader's real fold: per-store strata unions, then evaluation
        # on each merge.
        for digests in _fixture_digest_groups().values():
            self._assert_eval_parity([merge_tdigests_kway(digests)])

    def test_seeded_fuzz_parity(self):
        # Beyond the fixtures: duplicate means, exact (mean, weight) ties,
        # tiny and huge weights, negative and denormal-adjacent means.
        from zagg.stats import tdigest as zt

        rng = np.random.default_rng(29)
        checked = 0
        for _trial in range(60):
            k = int(rng.integers(2, 7))
            digests = []
            for _ in range(k):
                n = int(rng.integers(1, 80))
                means = rng.choice(
                    np.concatenate(
                        [
                            rng.normal(scale=100.0, size=n),
                            np.asarray([0.0, 1e-38, -1e-38, 5.0, 5.0]),
                        ]
                    ),
                    size=n,
                )
                means = np.sort(means.astype(np.float32).astype(np.float64))
                weights = rng.choice(
                    np.asarray([1.0, 2.0, 3.0, 1e-3, 1e6]), size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1]
                )
                digests.append(_digest(np.column_stack([means, weights])))
            delta = int(rng.choice([4, 32, DEFAULT_DELTA]))
            native = merge_tdigests_kway(digests, delta=delta)
            ref = zt.merge_tdigests_kway(digests, delta=delta)
            np.testing.assert_array_equal(native, ref)
            self._assert_eval_parity([native] + digests)
            checked += 1 + len(digests)
        assert checked >= 60


class TestReaderWithoutZagg:
    def test_surfaces_and_kernels_work_with_zagg_masked(self, tmp_path):
        """The issue #64 acceptance: the reader decodes with zagg ABSENT.

        A subprocess masks zagg via an import hook (raising on any
        ``zagg``/``zagg.*`` import — stronger than uninstalling, since a
        stray lazy import fails loudly), then runs the previously
        zagg-gated paths end to end: ``open_surface`` on the kitchen_sink
        strata fixture, ``read_tensors`` on the strata leaf, and the
        kernels themselves.
        """
        kitchen = DATA / "spec" / "kitchen_sink"
        leaf = DATA / "strata_hive" / "4" / "3" / "3" / "1" / "4" / "43314.zarr"
        script = textwrap.dedent(f"""
            import sys

            class _Mask:
                def find_spec(self, name, path=None, target=None):
                    if name == "zagg" or name.startswith("zagg."):
                        raise ImportError(f"masked for the no-zagg proof: {{name}}")
                    return None

            sys.meta_path.insert(0, _Mask())

            import numpy as np
            import moczarr
            from zarr.storage import LocalStore

            surf = moczarr.open_surface({str(kitchen)!r}, 6,
                                        ("h_tdigest_signal", "h_tdigest_noise"))
            vals = surf["h_tdigest"].values
            assert vals.shape[0] == 5 and vals.shape[1] > 0
            assert np.isfinite(vals).any()

            from moczarr.hhdc import read_tensors
            blocks = list(read_tensors(LocalStore({str(leaf)!r}), "6/h_tdigest_signal"))
            assert blocks and blocks[0][0].ndim == 3

            from moczarr.tdigest import (cdf_from_tdigest, merge_tdigests_kway,
                                         quantile_from_tdigest)
            d = np.asarray([[1.0, 1.0], [3.0, 2.0]], dtype=np.float32)
            assert quantile_from_tdigest(d, 1.0) == 3.0
            assert cdf_from_tdigest(d, 4.0) == 3.0
            assert merge_tdigests_kway([d, d]).shape[1] == 2

            assert not any(m == "zagg" or m.startswith("zagg.") for m in sys.modules)
            print("NO-ZAGG-OK")
        """)
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
        )
        assert proc.returncode == 0, proc.stderr
        assert "NO-ZAGG-OK" in proc.stdout

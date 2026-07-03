"""Benchmarks for kuant.sindy — null-testing and system-identification tools."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.sindy import permtest, sindylasso

# Optional heavy dependencies used by some tools.
_HAS_STATSMODELS = True
try:
    import statsmodels  # noqa: F401

    from kuant.sindy import grangerscan
except ImportError:
    _HAS_STATSMODELS = False

_HAS_SKLEARN = True
try:
    from kuant.sindy import pinnscan
except ImportError:
    _HAS_SKLEARN = False


def _corr_metric(x, y):
    return abs(float(np.corrcoef(x, y)[0, 1]))


# ---------- permtest ------------------------------------------------------


def test_bench_permtest_500x200(benchmark, rng=np.random.default_rng(0)):
    """Universal null-test with a light metric."""
    n = 500
    x = rng.normal(size=n)
    y = 0.4 * x + rng.normal(size=n)
    real = _corr_metric(x, y)
    benchmark(
        permtest,
        real,
        _corr_metric,
        x,
        y,
        n_perms=200,
        seed=0,
        higher_is_better=True,
    )


# ---------- sindylasso ----------------------------------------------------


def test_bench_sindylasso_20feat(benchmark, rng=np.random.default_rng(0)):
    """LASSO scan over a 20-feature library, 1000 samples."""
    n = 1_000
    library = {f"x{i}": rng.standard_normal(n) for i in range(20)}
    target = 0.4 * library["x0"] + rng.standard_normal(n)
    benchmark(sindylasso, target, library, n_splits=5)


# ---------- grangerscan --------------------------------------------------


@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels not installed")
def test_bench_grangerscan_5cand_3horizons(benchmark, rng=np.random.default_rng(0)):
    n = 500
    y = rng.normal(size=n)
    candidates = {f"c{i}": rng.normal(size=n) for i in range(5)}
    benchmark(grangerscan, y, candidates, horizons=[1, 2, 5])


# ---------- pinnscan ------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
def test_bench_pinnscan_10feat_50perms(benchmark, rng=np.random.default_rng(0)):
    n = 500
    library = {f"x{i}": rng.standard_normal(n) for i in range(10)}
    target = 0.4 * library["x0"] + rng.standard_normal(n)
    benchmark(
        pinnscan,
        target,
        library,
        n_splits=3,
        n_perms=50,
        n_estimators=50,
        random_state=0,
    )

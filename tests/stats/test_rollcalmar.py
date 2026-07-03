"""Test suite for kuant.stats.rollcalmar."""

from __future__ import annotations

import numpy as np

from kuant.stats import rollcalmar


def test_matches_manual_construction(rng):
    r = rng.normal(0.001, 0.01, 500)
    result = rollcalmar(r, window=252, ann_factor=252)

    # Manual: mean return * ann_factor / |MDD|
    window = r[-252:]
    equity = np.cumprod(1 + window)
    peak = np.maximum.accumulate(equity)
    mdd = float((equity / peak - 1).min())
    expected = np.mean(window) * 252 / abs(mdd)
    assert abs(result[-1] - expected) < 1e-12


def test_warmup_nan():
    r = np.random.default_rng(0).normal(size=500)
    result = rollcalmar(r, window=252)
    assert np.all(np.isnan(result[:251]))


def test_zero_drawdown_returns_nan():
    """Monotonically increasing series → MDD=0 → undefined."""
    r = np.full(100, 0.01)
    result = rollcalmar(r, window=50)
    assert np.all(np.isnan(result[49:]))


def test_positive_when_mean_positive(rng):
    """Positive mean returns + real drawdown → positive Calmar."""
    r = rng.normal(0.005, 0.01, 500)  # bias positive
    result = rollcalmar(r, window=252)
    finite = result[~np.isnan(result)]
    # At least most windows should have positive Calmar
    assert np.mean(finite > 0) > 0.5


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    r = rng.normal(0.001, 0.01, 300)
    r_cpu = rollcalmar(r, window=100)
    r_gpu = cp.asnumpy(rollcalmar(cp.asarray(r), window=100))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12, equal_nan=True)

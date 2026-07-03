"""Test suite for kuant.stats.rollmdd."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import rollmdd


def _manual_mdd(r):
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def test_matches_manual_last_window(rng):
    r = rng.normal(0.001, 0.01, 500)
    result = rollmdd(r, window=252)
    assert abs(result[-1] - _manual_mdd(r[-252:])) < 1e-14


def test_known_drawdown():
    """Hand-computed: +5%, -10%, +2%, -15%, +3% → -0.15 at last bar w=3."""
    r = np.array([0.05, -0.10, 0.02, -0.15, 0.03])
    # Window at t=4 covers [0.02, -0.15, 0.03]
    #   equity = [1.02, 0.867, 0.893]
    #   peak =   [1.02, 1.02, 1.02]
    #   dd =     [0, -0.15, -0.124...]
    #   min = -0.15
    result = rollmdd(r, window=3)
    assert abs(result[-1] - (-0.15)) < 1e-12


def test_all_positive_returns_zero_mdd():
    """Monotonically increasing equity → drawdown is zero."""
    r = np.full(50, 0.01)
    result = rollmdd(r, window=20)
    # DD is always 0 when equity is monotonic
    finite = result[~np.isnan(result)]
    assert np.all(np.abs(finite) < 1e-14)


def test_warmup_nan():
    r = np.random.default_rng(0).normal(size=100)
    result = rollmdd(r, window=30)
    assert np.all(np.isnan(result[:29]))
    assert np.all(np.isfinite(result[29:]))


def test_window_zero_raises():
    with pytest.raises(ValueError):
        rollmdd(np.arange(10.0), 0)


def test_window_larger_than_series_all_nan():
    result = rollmdd(np.arange(5.0), 10)
    assert np.all(np.isnan(result))


def test_nan_in_window_propagates():
    r = np.array([0.05, np.nan, 0.02, -0.15, 0.03])
    result = rollmdd(r, window=3)
    assert np.isnan(result[2])
    assert np.isnan(result[3])
    # Window [0.02, -0.15, 0.03] → no NaN → computes
    assert np.isfinite(result[4])


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    r = rng.normal(0.001, 0.01, 300)
    r_cpu = rollmdd(r, window=100)
    r_gpu = cp.asnumpy(rollmdd(cp.asarray(r), window=100))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12, equal_nan=True)

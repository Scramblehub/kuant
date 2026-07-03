"""Test suite for kuant.stats.rollsharpe."""

from __future__ import annotations

import numpy as np

from kuant.stats import rollsharpe


def _manual_sharpe(r, w, ann=1.0, rf=0.0, ddof=1):
    mean = np.mean(r[-w:]) - rf
    std = np.std(r[-w:], ddof=ddof)
    if std == 0:
        return np.nan
    return mean / std * np.sqrt(ann)


def test_matches_manual_no_ann(rng):
    r = rng.normal(0.001, 0.01, 500)
    result = rollsharpe(r, window=252, ann_factor=1.0)
    assert abs(result[-1] - _manual_sharpe(r, 252, 1.0)) < 1e-12


def test_matches_manual_ann_252(rng):
    r = rng.normal(0.0005, 0.012, 500)
    result = rollsharpe(r, window=252, ann_factor=252)
    assert abs(result[-1] - _manual_sharpe(r, 252, 252)) < 1e-12


def test_matches_manual_with_rf(rng):
    r = rng.normal(0.0005, 0.01, 500)
    result = rollsharpe(r, window=252, ann_factor=252, rf=0.0001)
    assert abs(result[-1] - _manual_sharpe(r, 252, 252, rf=0.0001)) < 1e-12


def test_warmup_is_nan():
    r = np.random.default_rng(0).normal(size=500)
    result = rollsharpe(r, window=252, ann_factor=1.0)
    assert np.all(np.isnan(result[:251]))
    assert np.all(np.isfinite(result[251:]))


def test_zero_std_returns_nan():
    r = np.full(100, 0.001)
    result = rollsharpe(r, window=50, ann_factor=1.0)
    # Constant series → std=0 → NaN
    assert np.all(np.isnan(result[49:]))


def test_ddof_0():
    r = np.random.default_rng(0).normal(size=500)
    result_ddof0 = rollsharpe(r, window=252, ddof=0)
    result_ddof1 = rollsharpe(r, window=252, ddof=1)
    # ddof=0 uses population std (n divisor); ddof=1 uses sample (n-1). Ratio:
    # ratio = sqrt((n-1)/n)
    ratio = result_ddof1[-1] / result_ddof0[-1]
    expected = np.sqrt(251 / 252)
    assert abs(ratio - expected) < 1e-10


def test_nan_in_window_propagates():
    r = np.arange(10.0)
    r[5] = np.nan
    result = rollsharpe(r, window=3)
    # Windows including index 5 → NaN
    assert np.isnan(result[5])
    assert np.isnan(result[6])
    assert np.isnan(result[7])
    # Window [6, 7, 8] doesn't include NaN → finite
    assert np.isfinite(result[8])


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    r = rng.normal(0.001, 0.01, 500)
    r_cpu = rollsharpe(r, window=252, ann_factor=252)
    r_gpu = cp.asnumpy(rollsharpe(cp.asarray(r), window=252, ann_factor=252))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12, equal_nan=True)

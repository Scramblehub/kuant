"""Test suite for kuant.stats.rollsortino."""

from __future__ import annotations

import numpy as np

from kuant.stats import rollsortino


def _manual_sortino(r, w, ann=1.0, target=0.0):
    window = r[-w:]
    mean = np.mean(window) - target
    below = np.where(window < target, target - window, 0.0)
    dd = np.sqrt(np.mean(below**2))
    if dd == 0:
        return np.nan
    return mean / dd * np.sqrt(ann)


def test_matches_manual_no_ann(rng):
    r = rng.normal(0.001, 0.01, 500)
    result = rollsortino(r, window=252, ann_factor=1.0, target=0.0)
    assert abs(result[-1] - _manual_sortino(r, 252, 1.0, 0.0)) < 1e-12


def test_matches_manual_ann_252(rng):
    r = rng.normal(0.0005, 0.012, 500)
    result = rollsortino(r, window=252, ann_factor=252)
    assert abs(result[-1] - _manual_sortino(r, 252, 252)) < 1e-12


def test_target_shifts_result(rng):
    """Higher target → more downside → lower Sortino for positive-mean returns."""
    r = rng.normal(0.001, 0.01, 500)
    high_bar = rollsortino(r, window=252, target=0.005)[-1]
    low_bar = rollsortino(r, window=252, target=0.0)[-1]
    assert high_bar < low_bar


def test_no_downside_returns_nan():
    """Constant positive series → no returns below target → Sortino undefined."""
    r = np.full(300, 0.01)
    result = rollsortino(r, window=100, target=0.0)
    assert np.all(np.isnan(result[99:]))


def test_warmup_is_nan():
    r = np.random.default_rng(0).normal(size=500)
    result = rollsortino(r, window=252, ann_factor=1.0)
    assert np.all(np.isnan(result[:251]))


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    r = rng.normal(0.001, 0.01, 500)
    r_cpu = rollsortino(r, window=252, ann_factor=252)
    r_gpu = cp.asnumpy(rollsortino(cp.asarray(r), window=252, ann_factor=252))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12, equal_nan=True)

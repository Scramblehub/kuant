'''Test suite for kuant.qm.zenoscan.'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.qm import zenoscan


def _linear_fit(X, y):
    '''Manual OLS fit — avoids sklearn dep for this test.'''
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _linear_predict(model, X):
    return X @ model


def _corr_metric(y_true, y_pred):
    return {'corr': float(np.corrcoef(y_true, y_pred)[0, 1])}


def test_scan_produces_metric_per_frequency():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = X @ [0.5, -0.3] + rng.normal(scale=0.3, size=500)
    result = zenoscan(
        _linear_fit, _linear_predict, _corr_metric,
        X, y, retrain_freqs=[10, 50], train_window=100,
    )
    assert set(result.metrics.keys()) == {10, 50}
    for freq in [10, 50]:
        assert 'corr' in result.metrics[freq]


def test_retrain_count_decreases_with_frequency():
    '''Higher retrain frequency (more days between retrains) → fewer retrains.'''
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 2))
    y = X @ [0.5, -0.3] + rng.normal(scale=0.3, size=1000)
    result = zenoscan(
        _linear_fit, _linear_predict, _corr_metric,
        X, y, retrain_freqs=[10, 100, 500], train_window=200,
    )
    assert result.retrain_counts[10] > result.retrain_counts[100] > result.retrain_counts[500]


def test_summary_readable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = X @ [0.5, -0.3] + rng.normal(size=300)
    result = zenoscan(
        _linear_fit, _linear_predict, _corr_metric,
        X, y, retrain_freqs=[20, 100], train_window=100,
    )
    text = result.summary()
    assert 'Zeno' in text
    assert 'corr' in text or 'Freq' in text

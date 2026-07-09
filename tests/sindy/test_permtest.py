"""Test suite for kuant.sindy.permtest."""

from __future__ import annotations

import numpy as np

from kuant.sindy import permtest


def _corr_metric(x, y):
    """Absolute Pearson correlation."""
    return abs(np.corrcoef(x.ravel(), y)[0, 1])


def test_signal_recovers_low_p_value():
    """y linearly caused by x → permutation p should be low."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 1))
    y = 0.5 * x.ravel() + rng.normal(scale=0.3, size=500)
    real = _corr_metric(x, y)
    result = permtest(real, _corr_metric, x, y, n_perms=200, seed=42)
    assert result.p_value < 0.01
    assert result.at_least_as_extreme < 5


def test_noise_gives_high_p_value():
    """y independent of x → p should be near 0.5."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 1))
    y = rng.normal(size=500)  # independent
    real = _corr_metric(x, y)
    result = permtest(real, _corr_metric, x, y, n_perms=200, seed=42)
    # Uncorrelated data: p should be in [0.1, 0.9]
    assert 0.05 < result.p_value < 0.95


def test_p_value_never_zero():
    """Even a perfect signal returns p >= 1/(n+1) due to +1 correction."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 1))
    y = x.ravel()  # perfect linear
    real = _corr_metric(x, y)
    result = permtest(real, _corr_metric, x, y, n_perms=100, seed=0)
    assert result.p_value >= 1 / 101


def test_higher_is_better_flag():
    """For a metric where lower is better, flip the flag."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    y = 0.3 * x.ravel() + rng.normal(scale=0.5, size=200)

    def mse(x, y):
        return float(np.mean((x.ravel() - y) ** 2))

    real = mse(x, y)
    result = permtest(real, mse, x, y, n_perms=100, higher_is_better=False)
    # Signal reduces MSE below random; p should be low
    assert result.p_value < 0.1


def test_summary_readable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 1))
    y = 0.3 * x.ravel() + rng.normal(size=100)
    real = _corr_metric(x, y)
    result = permtest(real, _corr_metric, x, y, n_perms=50)
    text = result.summary()
    assert "Permutation test" in text
    assert "p-value" in text

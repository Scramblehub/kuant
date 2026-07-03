"""Test suite for kuant.stats.dfa."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import dfa


def test_white_noise_h_near_half(rng):
    """iid Gaussian → H ~ 0.5."""
    r = rng.standard_normal(5000)
    result = dfa(r)
    assert abs(result.H - 0.5) < 0.1


def test_integrated_noise_h_near_1_5(rng):
    """cumsum of Gaussian → H ~ 1.5 (integrated random walk)."""
    r = rng.standard_normal(5000).cumsum()
    result = dfa(r)
    assert 1.35 < result.H < 1.65


def test_persistent_ar1_h_above_half(rng):
    """AR(1) with strong positive autocorrelation → H > 0.5."""
    n = 8000
    phi = 0.7
    e = rng.standard_normal(n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    result = dfa(r)
    assert result.H > 0.55


def test_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        dfa(np.zeros(30), min_w=10)


def test_result_has_expected_fields(rng):
    result = dfa(rng.standard_normal(2000))
    assert np.isfinite(result.H)
    assert len(result.windows) == result.n_windows
    assert result.log_F.shape == (result.n_windows,)


def test_summary_returns_string(rng):
    result = dfa(rng.standard_normal(2000))
    s = result.summary()
    assert isinstance(s, str)
    assert "DFA" in s


def test_scale_invariance(rng):
    """DFA exponent is invariant to input scale."""
    r = rng.standard_normal(3000)
    h1 = dfa(r).H
    h2 = dfa(r * 100).H
    assert abs(h1 - h2) < 1e-10

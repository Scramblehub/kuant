"""Test suite for kuant.stats.rolltailindex."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import rolltailindex


def test_warmup_is_nan(rng):
    x = rng.pareto(2.0, size=500) + 1
    result = rolltailindex(x, window=200)
    assert np.all(np.isnan(result[:199]))
    assert np.all(np.isfinite(result[199:]))


def test_recovers_stationary_pareto(rng):
    """A long stationary Pareto series has roughly constant rolling xi."""
    xi_true = 0.5
    sample = (1 - rng.uniform(size=5000)) ** (-xi_true)
    xi_t = rolltailindex(sample, window=1000, k_frac=0.10)
    finite = xi_t[np.isfinite(xi_t)]
    # Median should be near true
    assert abs(float(np.median(finite)) - xi_true) < 0.1


def test_window_zero_raises():
    with pytest.raises(ValueError):
        rolltailindex(np.arange(100.0), 0)


def test_shape():
    x = np.arange(100.0)
    assert rolltailindex(x, window=50).shape == (100,)

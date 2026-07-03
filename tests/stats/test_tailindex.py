"""Test suite for kuant.stats.tailindex."""

from __future__ import annotations

import numpy as np

from kuant.stats import tailindex


def test_pareto_recovery(rng):
    """Hill estimator recovers the true tail index of a Pareto sample."""
    # Standard Pareto with shape (1/xi) = 2 → xi = 0.5
    xi_true = 0.5
    sample = (1 - rng.uniform(size=20000)) ** (-xi_true)
    xi_hat = tailindex(sample, k_frac=0.05)
    assert abs(xi_hat - xi_true) < 0.05


def test_lighter_tail_gives_smaller_xi(rng):
    """A heavier-tailed sample should yield a larger Hill estimate.

    The Hill estimator is biased on non-Pareto distributions (well-known),
    but its ranking is meaningful: heavier tail → larger xi.
    """
    # Compare Pareto shape=1 (heavier) vs Pareto shape=3 (lighter)
    heavy = (1 - rng.uniform(size=10000)) ** (-1.0)
    light = (1 - rng.uniform(size=10000)) ** (-0.3)
    xi_heavy = tailindex(heavy, k_frac=0.05)
    xi_light = tailindex(light, k_frac=0.05)
    assert xi_heavy > xi_light + 0.3


def test_short_series_returns_nan():
    x = np.array([1.0, 2.0, 3.0])  # only 3 values
    assert np.isnan(tailindex(x, min_k=10))


def test_negative_and_nan_are_filtered():
    x = np.array([1.0, -2.0, 3.0, np.nan, 4.0, -5.0])
    # Only [1, 3, 4] are valid → 3 samples, too few for default min_k=10
    result = tailindex(x, min_k=10)
    assert np.isnan(result)


def test_k_frac_scales(rng):
    """Smaller k_frac → tighter tail focus → xi may shift."""
    sample = (1 - rng.uniform(size=10000)) ** (-0.7)
    xi_wide = tailindex(sample, k_frac=0.10)
    xi_narrow = tailindex(sample, k_frac=0.01, min_k=20)
    # Both should be finite and in the right ballpark
    assert np.isfinite(xi_wide)
    assert np.isfinite(xi_narrow)
    assert abs(xi_wide - 0.7) < 0.15
    assert abs(xi_narrow - 0.7) < 0.30

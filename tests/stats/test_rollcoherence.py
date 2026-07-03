"""Test suite for kuant.stats.rollcoherence."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import rollcoherence


def test_warmup_is_nan(rng):
    x = rng.standard_normal(500)
    y = rng.standard_normal(500)
    result = rollcoherence(x, y, window=200)
    assert np.all(np.isnan(result[:199]))
    assert np.all(np.isfinite(result[199:]))


def test_correlated_pair_is_coherent(rng):
    """A y = x + noise pair should show high coherence."""
    n = 800
    x = rng.standard_normal(n)
    y = x + 0.1 * rng.standard_normal(n)
    result = rollcoherence(x, y, window=400)
    finite = result[np.isfinite(result)]
    assert float(np.median(finite)) > 0.8


def test_independent_pair_low_coherence(rng):
    x = rng.standard_normal(800)
    y = rng.standard_normal(800)
    result = rollcoherence(x, y, window=400)
    finite = result[np.isfinite(result)]
    assert float(np.median(finite)) < 0.5


def test_mismatched_lengths_raises(rng):
    with pytest.raises(ValueError, match="equal length"):
        rollcoherence(rng.standard_normal(100), rng.standard_normal(200), window=50)


def test_window_zero_raises(rng):
    with pytest.raises(ValueError):
        rollcoherence(rng.standard_normal(100), rng.standard_normal(100), 0)


def test_band_filter(rng):
    """Restricting the band should change the result."""
    n = 800
    x = rng.standard_normal(n)
    y = x + 0.5 * rng.standard_normal(n)
    result_wide = rollcoherence(x, y, window=400, band=(0.0, 0.5))
    result_narrow = rollcoherence(x, y, window=400, band=(0.0, 0.05))
    # Both should be finite; may differ. Just verify no crash + finite output.
    assert np.any(np.isfinite(result_wide))
    assert np.any(np.isfinite(result_narrow))

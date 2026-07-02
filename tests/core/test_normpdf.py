"""Test suite for kuant.core.normpdf.

Covers: golden values, scipy match, edge cases, symmetry, backend parity.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import normpdf


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0.3989422804014327),   # peak
        (1.0, 0.24197072451914337),
        (-1.0, 0.24197072451914337),  # symmetric
        (2.0, 0.05399096651318806),
        (-2.0, 0.05399096651318806),
        (3.0, 0.0044318484119380075),
        (5.0, 1.4867195147342979e-06),
    ],
)
def test_golden_values(x, expected):
    result = normpdf(x)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, abs=1e-15)


def test_matches_scipy(rng):
    x = rng.uniform(-6, 6, size=10_000)
    result = normpdf(x)
    np.testing.assert_allclose(result, norm.pdf(x), atol=1e-15, rtol=1e-12)


def test_nan_passthrough():
    assert np.isnan(normpdf(float("nan")))


def test_positive_infinity():
    """phi(+inf) = 0."""
    assert normpdf(float("inf")) == 0.0


def test_negative_infinity():
    """phi(-inf) = 0."""
    assert normpdf(float("-inf")) == 0.0


def test_symmetry(rng):
    x = rng.uniform(-6, 6, size=1_000)
    np.testing.assert_allclose(normpdf(-x), normpdf(x), atol=1e-15)


def test_empty_array():
    result = normpdf(np.array([], dtype=np.float64))
    assert result.size == 0


def test_scalar_int_input():
    assert normpdf(0) == pytest.approx(0.3989422804014327, abs=1e-15)


def test_2d_array_preserves_shape(rng):
    x = rng.uniform(-3, 3, size=(4, 5))
    result = normpdf(x)
    assert result.shape == (4, 5)
    np.testing.assert_allclose(result, norm.pdf(x), atol=1e-15)


def test_dtype_preserved_float32(rng):
    x = rng.uniform(-3, 3, size=100).astype(np.float32)
    result = normpdf(x)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, norm.pdf(x), atol=1e-6)


def test_output_in_valid_range(rng):
    """phi(x) in [0, 1/sqrt(2*pi)]."""
    x = rng.uniform(-10, 10, size=1000)
    result = normpdf(x)
    peak = 1.0 / np.sqrt(2 * np.pi)
    assert np.all(result >= 0.0)
    assert np.all(result <= peak + 1e-15)


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x_cpu = rng.uniform(-4, 4, size=10_000)
    x_gpu = cp.asarray(x_cpu)
    np.testing.assert_allclose(normpdf(x_cpu), cp.asnumpy(normpdf(x_gpu)), atol=1e-14)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = normpdf(cp.asarray([0.0, 1.0]))
    assert isinstance(result, cp.ndarray)

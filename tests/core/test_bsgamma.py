"""Test suite for kuant.core.bsgamma."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bsput, bsputdelta, bsgamma


def _reference_gamma(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, 0.018762017345846895),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, 0.006813597181997181),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, 0.017036921138505086),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, 0.018974281789762865),
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    result = bsgamma(S, K, T, r, sigma, q)
    assert result == pytest.approx(expected, abs=1e-12)


def test_matches_reference_uniform(rng):
    n = 1000
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)
    np.testing.assert_allclose(
        bsgamma(S, K, T, r, sigma, q),
        _reference_gamma(S, K, T, r, sigma, q),
        atol=1e-12, rtol=1e-12,
    )


def test_gamma_nonneg(rng):
    """Gamma is always >= 0."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    gammas = bsgamma(S, K, T, r, sigma)
    assert np.all(gammas >= 0.0)


def test_gamma_peaks_near_atm():
    """Gamma should peak near ATM for a given tenor."""
    strikes = np.linspace(80, 120, 200)
    gammas = bsgamma(100.0, strikes, 0.25, 0.05, 0.20)
    peak_idx = int(np.argmax(gammas))
    # Peak should be near K=100 (within ~5% for a 3-month put)
    assert 90 < strikes[peak_idx] < 110


def test_gamma_matches_finite_difference_of_delta(rng):
    """gamma ~ (delta(S+h) - delta(S-h)) / (2h). Cross-check against bsputdelta."""
    n = 20
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.2, 1.5, size=n)
    r, sigma = 0.05, 0.25
    h = 1e-3
    analytic = bsgamma(S, K, T, r, sigma)
    fd = (bsputdelta(S + h, K, T, r, sigma) - bsputdelta(S - h, K, T, r, sigma)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)


def test_gamma_matches_second_derivative_of_price(rng):
    """gamma ~ (P(S+h) - 2*P(S) + P(S-h)) / h^2. Cross-check against bsput.

    Second-order central diff needs bigger h to overcome noise.
    """
    n = 20
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.2, 1.5, size=n)
    r, sigma = 0.05, 0.25
    h = 1e-2
    analytic = bsgamma(S, K, T, r, sigma)
    fd = (bsput(S + h, K, T, r, sigma) - 2 * bsput(S, K, T, r, sigma) + bsput(S - h, K, T, r, sigma)) / (h * h)
    np.testing.assert_allclose(analytic, fd, atol=1e-4)


# Edge cases
def test_expired(): assert bsgamma(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0
def test_zero_vol(): assert bsgamma(100.0, 100.0, 1.0, 0.05, 0.0) == 0.0
def test_zero_spot(): assert bsgamma(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0
def test_zero_strike(): assert bsgamma(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0
def test_nan_passthrough(): assert np.isnan(bsgamma(float("nan"), 100.0, 1.0, 0.05, 0.20))


def test_dtype_preserved_float32():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [np.array([1.0], dtype=np.float32),
                                                        np.array([0.05], dtype=np.float32),
                                                        np.array([0.2], dtype=np.float32)]
    result = bsgamma(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    r_cpu = bsgamma(S, K, T, r, sigma)
    r_gpu = bsgamma(cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma))
    np.testing.assert_allclose(r_cpu, cp.asnumpy(r_gpu), atol=1e-10)

"""Test suite for kuant.core.bsvega."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bsput, bsvega


def _reference_vega(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, 37.52403469169379),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, 13.627194363994361),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, 34.07384227701017),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, 37.94856357952573),
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    result = bsvega(S, K, T, r, sigma, q)
    assert result == pytest.approx(expected, abs=1e-10)


def test_matches_reference_uniform(rng):
    n = 1000
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)
    np.testing.assert_allclose(
        bsvega(S, K, T, r, sigma, q),
        _reference_vega(S, K, T, r, sigma, q),
        atol=1e-10, rtol=1e-12,
    )


def test_vega_nonneg(rng):
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    vegas = bsvega(S, K, T, r, sigma)
    assert np.all(vegas >= 0.0)


def test_vega_matches_finite_difference_of_price(rng):
    """vega ~ (P(sigma+h) - P(sigma-h)) / (2h). Cross-check against bsput."""
    n = 20
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.2, 1.5, size=n)
    r, sigma = 0.05, 0.25
    h = 1e-4
    analytic = bsvega(S, K, T, r, sigma)
    fd = (bsput(S, K, T, r, sigma + h) - bsput(S, K, T, r, sigma - h)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, atol=1e-6)


def test_vega_peaks_near_atm():
    """Vega should peak near ATM for a given tenor."""
    strikes = np.linspace(50, 150, 200)
    vegas = bsvega(100.0, strikes, 1.0, 0.05, 0.20)
    peak_idx = int(np.argmax(vegas))
    # For 1y ATM-forward, peak is slightly above spot; broad range OK.
    assert 90 < strikes[peak_idx] < 130


# Edge cases
def test_expired(): assert bsvega(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0
def test_zero_vol(): assert bsvega(100.0, 100.0, 1.0, 0.05, 0.0) == 0.0
def test_zero_spot(): assert bsvega(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0
def test_zero_strike(): assert bsvega(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0
def test_nan_passthrough(): assert np.isnan(bsvega(float("nan"), 100.0, 1.0, 0.05, 0.20))


def test_dtype_preserved_float32():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [np.array([1.0], dtype=np.float32),
                                                        np.array([0.05], dtype=np.float32),
                                                        np.array([0.2], dtype=np.float32)]
    result = bsvega(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    r_cpu = bsvega(S, K, T, r, sigma)
    r_gpu = bsvega(cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma))
    np.testing.assert_allclose(r_cpu, cp.asnumpy(r_gpu), atol=1e-8)

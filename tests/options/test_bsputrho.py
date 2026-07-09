"""Test suite for kuant.core.bsputrho."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bsput
from kuant.options import bsputrho


def _reference_rho(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -T * K * np.exp(-r * T) * norm.cdf(-d2)


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, -41.89046090469506),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, -7.823449137492057),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, -88.67584456613379),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, -47.5614712250357),
        (50.0, 100.0, 0.5, 0.05, 0.30, 0.0, -48.737919577502964),  # deep ITM
        (150.0, 100.0, 0.5, 0.05, 0.30, 0.0, -1.3278651923734606),  # deep OTM
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    result = bsputrho(S, K, T, r, sigma, q)
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
        bsputrho(S, K, T, r, sigma, q),
        _reference_rho(S, K, T, r, sigma, q),
        atol=1e-10,
        rtol=1e-12,
    )


def test_rho_nonpositive(rng):
    """Put rho always <= 0."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    rhos = bsputrho(S, K, T, r, sigma)
    assert np.all(rhos <= 1e-12)


def test_rho_monotonic_in_strike():
    """Higher K -> more negative rho (more strike to discount)."""
    strikes = np.linspace(50, 150, 100)
    rhos = bsputrho(100.0, strikes, 1.0, 0.05, 0.20)
    assert np.all(np.diff(rhos) < 0)


def test_rho_matches_finite_difference_of_price(rng):
    """rho ~ (P(r+h) - P(r-h)) / (2h). Cross-check against bsput."""
    n = 20
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.2, 1.5, size=n)
    r, sigma = 0.05, 0.25
    h = 1e-5
    analytic = bsputrho(S, K, T, r, sigma)
    fd = (bsput(S, K, T, r + h, sigma) - bsput(S, K, T, r - h, sigma)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, atol=1e-6)


# Edge cases
def test_expired():
    assert bsputrho(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_zero_vol_exercise():
    """sigma=0, K exercises for sure -> rho = -T*K*exp(-r*T)."""
    # S=100, K=110, T=1, r=0.05: 110*e^-0.05 = 104.63 > 100 -> exercise
    result = bsputrho(100.0, 110.0, 1.0, 0.05, 0.0)
    expected = -1.0 * 110.0 * np.exp(-0.05)
    assert result == pytest.approx(expected, abs=1e-12)


def test_zero_vol_no_exercise():
    """sigma=0, K doesn't exercise -> rho = 0."""
    assert bsputrho(100.0, 90.0, 1.0, 0.05, 0.0) == 0.0


def test_zero_spot():
    """S=0 -> put worth K*exp(-r*T), rho = -T*K*exp(-r*T)."""
    result = bsputrho(0.0, 100.0, 1.0, 0.05, 0.20)
    expected = -1.0 * 100.0 * np.exp(-0.05)
    assert result == pytest.approx(expected, abs=1e-12)


def test_zero_strike():
    assert bsputrho(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0


def test_nan_passthrough():
    assert np.isnan(bsputrho(float("nan"), 100.0, 1.0, 0.05, 0.20))


def test_dtype_preserved_float32():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    result = bsputrho(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    r_cpu = bsputrho(S, K, T, r, sigma)
    r_gpu = bsputrho(cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma))
    np.testing.assert_allclose(r_cpu, cp.asnumpy(r_gpu), atol=1e-8)

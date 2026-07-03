"""Test suite for kuant.core.bscallrho.

Includes put-call parity for rho: rho_call - rho_put = T * K * exp(-r*T).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bscall
from kuant.options import bscallrho, bsputrho


def _reference_rho(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return T * K * np.exp(-r * T) * norm.cdf(d2)


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, 53.232481545376345),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, 68.27490482256506),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, 25.4716863739519),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, 47.5614712250357),
        (150.0, 100.0, 0.5, 0.05, 0.30, 0.0, 47.437630409043166),   # deep ITM
        (50.0, 100.0, 0.5, 0.05, 0.30, 0.0, 0.027576023913668958),  # deep OTM
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    result = bscallrho(S, K, T, r, sigma, q)
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
        bscallrho(S, K, T, r, sigma, q),
        _reference_rho(S, K, T, r, sigma, q),
        atol=1e-10, rtol=1e-12,
    )


def test_put_call_parity_for_rho(rng):
    """rho_call - rho_put = T * K * exp(-r*T). Machine-precision identity."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)

    rc = bscallrho(S, K, T, r, sigma)
    rp = bsputrho(S, K, T, r, sigma)
    np.testing.assert_allclose(rc - rp, T * K * np.exp(-r * T), atol=1e-10)


def test_rho_nonneg(rng):
    """Call rho always >= 0."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    rhos = bscallrho(S, K, T, r, sigma)
    assert np.all(rhos >= -1e-12)


def test_rho_monotonic_in_spot():
    """Higher S -> higher call rho (more likely to exercise -> discounted strike matters more).

    Note: call rho is NOT monotonic in K (unlike put rho). Deep ITM:
    rho ~ T*K*exp(-r*T) grows linearly in K. Deep OTM: rho -> 0. So it peaks
    somewhere in between. K-monotonicity fails; S-monotonicity holds.
    """
    spots = np.linspace(50, 150, 100)
    rhos = bscallrho(spots, 100.0, 1.0, 0.05, 0.20)
    assert np.all(np.diff(rhos) > 0)


def test_rho_matches_finite_difference_of_price(rng):
    """rho ~ (bscall(r+h) - bscall(r-h)) / (2h)."""
    n = 20
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.2, 1.5, size=n)
    r, sigma = 0.05, 0.25
    h = 1e-5
    analytic = bscallrho(S, K, T, r, sigma)
    fd = (bscall(S, K, T, r + h, sigma) - bscall(S, K, T, r - h, sigma)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, atol=1e-6)


# Edge cases
def test_expired(): assert bscallrho(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_zero_vol_exercise():
    """sigma=0, exercises -> rho = T*K*exp(-r*T)."""
    # S=100, K=90, T=1, r=0.05, q=0: 100 > 90*e^-0.05=85.6 -> exercise
    result = bscallrho(100.0, 90.0, 1.0, 0.05, 0.0)
    expected = 1.0 * 90.0 * np.exp(-0.05)
    assert result == pytest.approx(expected, abs=1e-12)


def test_zero_vol_no_exercise():
    """sigma=0, no exercise -> rho = 0."""
    assert bscallrho(100.0, 150.0, 1.0, 0.05, 0.0) == 0.0


def test_zero_spot(): assert bscallrho(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0
def test_zero_strike(): assert bscallrho(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0
def test_nan_passthrough(): assert np.isnan(bscallrho(float("nan"), 100.0, 1.0, 0.05, 0.20))


def test_dtype_preserved_float32():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [np.array([1.0], dtype=np.float32),
                                                        np.array([0.05], dtype=np.float32),
                                                        np.array([0.2], dtype=np.float32)]
    result = bscallrho(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    r_cpu = bscallrho(S, K, T, r, sigma)
    r_gpu = bscallrho(cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma))
    np.testing.assert_allclose(r_cpu, cp.asnumpy(r_gpu), atol=1e-8)

"""Test suite for kuant.core.bscall.

Includes the strongest cross-check: put-call parity against bsput.
If either kernel drifts, parity fails to machine precision.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bscall, bsput


def _reference_call(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, 10.450583572185565),   # ATM
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, 24.58883544392775),     # ITM (S > K)
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, 3.2474774165608125),   # OTM (K > S)
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, 8.652528553942709),   # with dividend
        (42.0, 40.0, 0.5, 0.10, 0.20, 0.0, 4.759422392871532),      # Hull textbook
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    result = bscall(S, K, T, r, sigma, q)
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
        bscall(S, K, T, r, sigma, q),
        _reference_call(S, K, T, r, sigma, q),
        atol=1e-12, rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Put-Call Parity — the strongest cross-check
# ---------------------------------------------------------------------------


def test_put_call_parity_atm():
    """C - P = S*exp(-q*T) - K*exp(-r*T). ATM 1y."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    C = bscall(S, K, T, r, sigma)
    P = bsput(S, K, T, r, sigma)
    expected_diff = S - K * np.exp(-r * T)
    assert (C - P) == pytest.approx(expected_diff, abs=1e-12)


def test_put_call_parity_random(rng):
    """Parity holds over 1000 random points to machine precision."""
    n = 1000
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)

    C = bscall(S, K, T, r, sigma, q)
    P = bsput(S, K, T, r, sigma, q)
    expected_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(C - P, expected_diff, atol=1e-12)


def test_broadcasting_strike_curve():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    result = bscall(100.0, strikes, 1.0, 0.05, 0.20)
    reference = _reference_call(100.0, strikes, 1.0, 0.05, 0.20)
    assert result.shape == (5,)
    np.testing.assert_allclose(result, reference, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_expired_itm():
    """T=0, S > K -> call = S - K."""
    assert bscall(100.0, 80.0, 0.0, 0.05, 0.20) == pytest.approx(20.0, abs=1e-12)


def test_expired_otm():
    """T=0, S < K -> call = 0."""
    assert bscall(80.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_expired_atm():
    """T=0, S == K -> call = 0."""
    assert bscall(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_zero_vol_exercise():
    """sigma=0 with forward > K -> call worth S*exp(-q*T) - K*exp(-r*T)."""
    # S=100, K=90, T=1, r=0.05, q=0: forward=100, K*e^-0.05=85.6, exercise
    result = bscall(100.0, 90.0, 1.0, 0.05, 0.0)
    expected = 100.0 - 90.0 * np.exp(-0.05)
    assert result == pytest.approx(expected, abs=1e-12)


def test_zero_vol_no_exercise():
    """sigma=0 with forward < K -> call worthless."""
    # S=100, K=110, T=1, r=0.05, q=0: 100 < 110*e^-0.05=104.63 -> worthless
    assert bscall(100.0, 110.0, 1.0, 0.05, 0.0) == 0.0


def test_zero_spot():
    """S=0 -> call worthless."""
    assert bscall(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0


def test_zero_strike():
    """K=0 -> guaranteed exercise, call worth S*exp(-q*T)."""
    result = bscall(100.0, 0.0, 1.0, 0.05, 0.20, 0.03)
    expected = 100.0 * np.exp(-0.03)
    assert result == pytest.approx(expected, abs=1e-12)


def test_nan_passthrough():
    assert np.isnan(bscall(float("nan"), 100.0, 1.0, 0.05, 0.20))
    assert np.isnan(bscall(100.0, float("nan"), 1.0, 0.05, 0.20))


def test_dtype_preserved_float32():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [np.array([1.0], dtype=np.float32),
                                                        np.array([0.05], dtype=np.float32),
                                                        np.array([0.2], dtype=np.float32)]
    result = bscall(*args)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_call_nonneg(rng):
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    prices = bscall(S, K, T, r, sigma)
    assert np.all(prices >= 0.0)


def test_call_monotonic_in_strike():
    """d(call)/dK < 0: higher K -> lower call price."""
    strikes = np.linspace(50, 150, 100)
    prices = bscall(100.0, strikes, 1.0, 0.05, 0.20)
    assert np.all(np.diff(prices) < 0)


def test_call_monotonic_in_vol():
    """d(call)/d(sigma) > 0 (vega > 0)."""
    sigmas = np.linspace(0.05, 0.80, 50)
    prices = bscall(100.0, 100.0, 1.0, 0.05, sigmas)
    assert np.all(np.diff(prices) > 0)


def test_call_bounded_above_by_spot(rng):
    """C <= S*exp(-q*T) (can't be worth more than the underlying)."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)
    prices = bscall(S, K, T, r, sigma, q)
    upper_bound = S * np.exp(-q * T)
    assert np.all(prices <= upper_bound + 1e-12)


# ---------------------------------------------------------------------------
# CPU == GPU
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    n = 1000
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    r_cpu = bscall(S, K, T, r, sigma)
    r_gpu = bscall(cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma))
    np.testing.assert_allclose(r_cpu, cp.asnumpy(r_gpu), atol=1e-10)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = bscall(cp.asarray([100.0]), 100.0, 1.0, 0.05, 0.20)
    assert isinstance(result, cp.ndarray)

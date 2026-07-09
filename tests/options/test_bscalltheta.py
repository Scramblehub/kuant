"""Test suite for kuant.options.bscalltheta.

Validation strategy:
  1. Match scipy analytic reference
  2. Match finite-difference of bscall
  3. Put-call parity for theta:
       theta_call - theta_put = q·S·e^(-q·T) - r·K·e^(-r·T)
  4. Edge cases: expired, zero vol, S=0, K=0, deep OTM/ITM
  5. Sign check: typical ATM call theta is negative
  6. Batched input
  7. dtype preservation
  8. CPU==GPU parity
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bscall
from kuant.options import bscalltheta, bsputtheta


def _ref_call_theta(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (
        -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
        + q * S * np.exp(-q * T) * norm.cdf(d1)
    )


# ---------------------------------------------------------------------------
# 1. Match scipy reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0),  # OTM call
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0),  # ITM call
        (100.0, 100.0, 0.25, 0.05, 0.20, 0.0),  # short T
        (100.0, 100.0, 2.0, 0.05, 0.30, 0.0),  # longer T, higher vol
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03),  # with dividend
        (100.0, 100.0, 1.0, 0.00, 0.20, 0.00),  # r=q=0
    ],
)
def test_matches_scipy_reference(S, K, T, r, sigma, q):
    assert abs(bscalltheta(S, K, T, r, sigma, q) - _ref_call_theta(S, K, T, r, sigma, q)) < 1e-10


# ---------------------------------------------------------------------------
# 2. Finite-difference check: theta = -d(price)/dT
# ---------------------------------------------------------------------------


def test_finite_difference_matches():
    """theta = -d(price)/dT. h=1e-5 balances truncation and roundoff."""
    S, K, r, sigma, q = 100.0, 100.0, 0.05, 0.20, 0.02
    for T in [0.25, 0.5, 1.0, 2.0]:
        dt = 1e-5
        num = -(bscall(S, K, T + dt, r, sigma, q) - bscall(S, K, T - dt, r, sigma, q)) / (2 * dt)
        ana = bscalltheta(S, K, T, r, sigma, q)
        assert abs(ana - num) < 1e-8


# ---------------------------------------------------------------------------
# 3. Put-call parity
# ---------------------------------------------------------------------------


def test_put_call_parity():
    """theta_call - theta_put = q·S·e^(-q·T) - r·K·e^(-r·T)"""
    for S, K, T, r, sigma, q in [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.02),
        (100.0, 120.0, 0.5, 0.03, 0.15, 0.01),
        (50.0, 75.0, 2.0, 0.04, 0.25, 0.03),
    ]:
        lhs = bscalltheta(S, K, T, r, sigma, q) - bsputtheta(S, K, T, r, sigma, q)
        rhs = q * S * np.exp(-q * T) - r * K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


def test_expired_zero():
    assert bscalltheta(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0
    assert bscalltheta(120.0, 100.0, 0.0, 0.05, 0.20) == 0.0  # ITM expired


def test_S_zero_gives_zero():
    assert bscalltheta(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0


def test_K_zero_gives_qS_growth():
    """When K=0 the call is worth S·e^(-q·T); theta = q·S·e^(-q·T)."""
    S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.03
    expected = q * S * np.exp(-q * T)
    assert abs(bscalltheta(S, 0.0, T, r, sigma, q) - expected) < 1e-10


def test_ATM_call_theta_negative():
    """Long ATM call decays with time -> theta negative."""
    assert bscalltheta(100.0, 100.0, 1.0, 0.05, 0.20) < 0


def test_short_tenor_theta_more_negative():
    """As T -> 0 for OTM, theta -> 0; for ATM it accelerates."""
    short = bscalltheta(100.0, 100.0, 0.1, 0.05, 0.20)
    long = bscalltheta(100.0, 100.0, 1.0, 0.05, 0.20)
    assert short < long  # more negative = "smaller"


def test_zero_vol_no_exercise_zero():
    # OTM at r=0 -> guaranteed no exercise -> theta 0
    result = bscalltheta(100.0, 200.0, 1.0, 0.0, 0.0)
    assert result == 0.0


# ---------------------------------------------------------------------------
# 5. Batched
# ---------------------------------------------------------------------------


def test_batched_matches_scalar():
    S_arr = np.array([90.0, 100.0, 110.0])
    K_arr = np.array([100.0, 100.0, 100.0])
    result = bscalltheta(S_arr, K_arr, 1.0, 0.05, 0.20)
    for i, S in enumerate([90.0, 100.0, 110.0]):
        assert abs(result[i] - bscalltheta(S, 100.0, 1.0, 0.05, 0.20)) < 1e-12


def test_broadcasting():
    S = np.array([80.0, 100.0, 120.0])
    K = 100.0
    result = bscalltheta(S, K, 1.0, 0.05, 0.20)
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# 6. dtype
# ---------------------------------------------------------------------------


def test_dtype_float32_preserved():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    result = bscalltheta(*args)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# 7. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    S = rng.uniform(50, 150, 100)
    K = rng.uniform(50, 150, 100)
    T = rng.uniform(0.1, 2.0, 100)
    r = rng.uniform(-0.01, 0.10, 100)
    sigma = rng.uniform(0.1, 0.6, 100)
    r_cpu = bscalltheta(S, K, T, r, sigma)
    r_gpu = cp.asnumpy(
        bscalltheta(cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma))
    )
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

"""Test suite for kuant.core.bsputdelta.

Same 5-layer validation strategy as bsput:
  1. Golden values      — scipy-derived (independent of our normcdf)
  2. Reference match    — 1000 uniform samples vs scipy.stats.norm.cdf
  3. Edge cases         — T=0, sigma=0, S=0, K=0, NaN, dtype
  4. Property tests     — bounded [-1, 0], monotonic in K, put spread
  5. CPU==GPU parity
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bsput, bsputdelta


# Independent reference (uses scipy directly, not our normcdf)
def _reference_delta(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return -np.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        # ATM 1y
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, -0.3631693488243809),
        # OTM (S > K)
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, -0.07136259733507186),
        # ITM (K > S)
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, -0.7128083620948729),
        # With dividend
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, -0.40830553575872397),
        # Deep ITM, short tenor — delta near -1
        (50.0, 100.0, 0.5, 0.05, 0.30, 0.0, -0.9988312101690474),
        # Deep OTM — delta near 0
        (150.0, 100.0, 0.5, 0.05, 0.30, 0.0, -0.016368338210787984),
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    result = bsputdelta(S, K, T, r, sigma, q)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Reference match
# ---------------------------------------------------------------------------


def test_matches_reference_uniform(rng):
    n = 1000
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)

    result = bsputdelta(S, K, T, r, sigma, q)
    reference = _reference_delta(S, K, T, r, sigma, q)
    np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)


def test_broadcasting_strike_curve():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    result = bsputdelta(100.0, strikes, 1.0, 0.05, 0.20)
    assert result.shape == (5,)
    # Delta is monotonically decreasing (more negative) as K increases.
    # (Higher strike -> more ITM -> more negative delta.)
    assert np.all(np.diff(result) < 0)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_expired_otm():
    """T=0, K < S -> delta = 0 (put worthless)."""
    assert bsputdelta(100.0, 80.0, 0.0, 0.05, 0.20) == 0.0


def test_expired_itm():
    """T=0, K > S -> delta = -1 (exercise for sure)."""
    assert bsputdelta(80.0, 100.0, 0.0, 0.05, 0.20) == -1.0


def test_expired_atm():
    """T=0, K == S -> delta = 0 by convention (right-derivative)."""
    assert bsputdelta(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_zero_vol_deterministic_exercise():
    """sigma=0, K*exp(-r*T) > S*exp(-q*T) -> guaranteed exercise, delta=-exp(-q*T)."""
    # S=100, K=110, T=1, r=0.05, q=0: 110*e^-0.05 = 104.63 > 100 -> exercise
    result = bsputdelta(100.0, 110.0, 1.0, 0.05, 0.0)
    expected = -np.exp(0.0)  # q=0 -> delta = -1
    assert result == pytest.approx(expected, abs=1e-12)


def test_zero_vol_no_exercise():
    """sigma=0, K*exp(-r*T) < S*exp(-q*T) -> put worthless, delta=0."""
    assert bsputdelta(100.0, 90.0, 1.0, 0.05, 0.0) == 0.0


def test_zero_vol_with_dividend():
    """sigma=0, T>0, with q>0: check the exp(-q*T) factor."""
    # S=100, K=200, T=1, r=0.05, sigma=0, q=0.03
    # 200*e^-0.05 = 190.25 > 100*e^-0.03 = 97.04 -> exercise
    # delta = -exp(-0.03) ~ -0.9704
    result = bsputdelta(100.0, 200.0, 1.0, 0.05, 0.0, 0.03)
    assert result == pytest.approx(-np.exp(-0.03), abs=1e-12)


def test_zero_spot():
    """S=0 -> put guaranteed exercise -> delta = -exp(-q*T)."""
    result = bsputdelta(0.0, 100.0, 1.0, 0.05, 0.20, 0.02)
    expected = -np.exp(-0.02)
    assert result == pytest.approx(expected, abs=1e-12)


def test_zero_strike():
    """K=0 -> put never exercises -> delta = 0."""
    assert bsputdelta(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0


def test_nan_passthrough():
    """NaN in any input -> NaN out."""
    assert np.isnan(bsputdelta(float("nan"), 100.0, 1.0, 0.05, 0.20))
    assert np.isnan(bsputdelta(100.0, float("nan"), 1.0, 0.05, 0.20))
    assert np.isnan(bsputdelta(100.0, 100.0, float("nan"), 0.05, 0.20))


def test_dtype_preserved_float32():
    """float32 in -> float32 out (q default doesn't promote)."""
    S = np.array([100.0], dtype=np.float32)
    K = np.array([100.0], dtype=np.float32)
    T = np.array([1.0], dtype=np.float32)
    r = np.array([0.05], dtype=np.float32)
    sigma = np.array([0.2], dtype=np.float32)
    result = bsputdelta(S, K, T, r, sigma)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(-0.3631693488243809, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_delta_in_range(rng):
    """Delta always in [-1, 0]."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)
    deltas = bsputdelta(S, K, T, r, sigma, q)
    assert np.all(deltas <= 0.0 + 1e-15)
    assert np.all(deltas >= -1.0 - 1e-15)


def test_delta_monotonic_in_strike():
    """Higher K -> more negative delta (put more ITM)."""
    strikes = np.linspace(50, 150, 100)
    deltas = bsputdelta(100.0, strikes, 1.0, 0.05, 0.20)
    assert np.all(np.diff(deltas) < 0)


def test_delta_monotonic_in_spot():
    """Higher S -> less negative delta (put less ITM)."""
    spots = np.linspace(50, 150, 100)
    deltas = bsputdelta(spots, 100.0, 1.0, 0.05, 0.20)
    assert np.all(np.diff(deltas) > 0)


def test_delta_matches_finite_difference(rng):
    """delta ~ (bsput(S+h) - bsput(S-h)) / (2h) — validates against bsput.

    This is the strongest property test: delta IS the S-derivative of bsput
    by definition. Bump-and-reprice must agree with the closed form.
    """
    n = 20
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.2, 1.5, size=n)
    r = 0.05
    sigma = 0.25
    h = 1e-4  # small bump

    analytic = bsputdelta(S, K, T, r, sigma)
    fd = (bsput(S + h, K, T, r, sigma) - bsput(S - h, K, T, r, sigma)) / (2 * h)

    # Central difference is O(h^2) accurate; 1e-4 bump gives ~1e-8 error
    np.testing.assert_allclose(analytic, fd, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. CPU == GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    n = 1000
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)

    result_cpu = bsputdelta(S, K, T, r, sigma)
    result_gpu = bsputdelta(
        cp.asarray(S), cp.asarray(K), cp.asarray(T),
        cp.asarray(r), cp.asarray(sigma),
    )
    np.testing.assert_allclose(result_cpu, cp.asnumpy(result_gpu), atol=1e-10)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = bsputdelta(cp.asarray([100.0]), 100.0, 1.0, 0.05, 0.20)
    assert isinstance(result, cp.ndarray)

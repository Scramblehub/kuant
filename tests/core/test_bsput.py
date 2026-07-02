"""Test suite for kuant.core.bsput.

Validation strategy mirrors normcdf's five-layer approach:

  1. Golden values      — hand-verified from Hull "Options, Futures, and Other
                          Derivatives" and other textbook examples
  2. Reference match    — independent BS implementation using scipy directly
  3. Edge cases         — T=0, sigma=0, S=0, K=0, NaN, broadcasting
  4. Property tests     — monotonicity, put-call parity, positivity
  5. CPU==GPU parity    — GPU output matches CPU output

The reference in test 2 uses scipy.stats.norm.cdf DIRECTLY (not our normcdf),
so if normcdf has a bug the bsput tests will catch it via composition.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bsput


# ---------------------------------------------------------------------------
# Independent reference implementation for validation
# ---------------------------------------------------------------------------


def _reference_bsput(S, K, T, r, sigma, q=0.0):
    """Reference BS put using scipy.stats.norm.cdf directly.

    This is what our test compares against — independent of kuant.core.normcdf,
    so composition bugs get caught.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# 1. Golden values — textbook examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        # Classic Hull textbook example (Ch 15, Example 15.6-ish):
        # S=42, K=40, T=0.5, r=0.10, sigma=0.20, q=0 -> put ~ 0.8086
        (42.0, 40.0, 0.5, 0.10, 0.20, 0.0, 0.8085993729000516),
        # ATM 1-year put: S=100, K=100, T=1, r=0.05, sigma=0.2, q=0
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, 5.573526022256971),
        # Deep OTM put: S=100, K=80, T=1, r=0.05, sigma=0.2, q=0
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, 0.6871894039848714),
        # Deep ITM put: S=100, K=120, T=1, r=0.05, sigma=0.2, q=0
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, 17.3950083566465),
        # With dividend: S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.03
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, 6.7309176491633025),
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    """Hand-verified BS put prices to 12 decimals."""
    result = bsput(S, K, T, r, sigma, q)
    assert isinstance(result, float), "scalar inputs -> scalar output"
    assert result == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# 2. Reference match — batched vs scipy-based reference
# ---------------------------------------------------------------------------


def test_matches_reference_uniform(rng):
    """1000 random parameter sets match independent reference impl."""
    n = 1000
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)

    result = bsput(S, K, T, r, sigma, q)
    reference = _reference_bsput(S, K, T, r, sigma, q)

    np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)


def test_broadcasting_strike_curve():
    """Scalar spot + array strikes -> curve output. Broadcasts correctly."""
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    result = bsput(100.0, strikes, 1.0, 0.05, 0.20)
    reference = _reference_bsput(100.0, strikes, 1.0, 0.05, 0.20)

    assert result.shape == (5,)
    np.testing.assert_allclose(result, reference, atol=1e-12)


def test_broadcasting_full_grid():
    """(N, M) strike x tenor grid via broadcasting."""
    strikes = np.array([90.0, 100.0, 110.0])[:, None]     # column
    tenors = np.array([0.25, 0.5, 1.0, 2.0])[None, :]     # row
    result = bsput(100.0, strikes, tenors, 0.05, 0.20)

    assert result.shape == (3, 4)

    # Spot check: at K=100, T=1 -> should be ~5.57
    assert result[1, 2] == pytest.approx(5.573526022256971, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_expired_option_intrinsic():
    """T = 0 -> put = max(K - S, 0)."""
    # OTM: K < S -> 0
    assert bsput(100.0, 80.0, 0.0, 0.05, 0.20) == 0.0
    # ITM: K > S -> K - S
    assert bsput(80.0, 100.0, 0.0, 0.05, 0.20) == pytest.approx(20.0, abs=1e-12)
    # ATM: K == S -> 0
    assert bsput(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_zero_vol_deterministic():
    """sigma = 0 -> discounted deterministic intrinsic."""
    # No vol, no dividends, positive rate: put worth max(K*exp(-r*T) - S, 0)
    # S=100, K=110, T=1, r=0.05, sigma=0 -> put = max(110*e^(-0.05) - 100, 0) = 4.634...
    price = bsput(100.0, 110.0, 1.0, 0.05, 0.0)
    expected = 110.0 * np.exp(-0.05) - 100.0
    assert price == pytest.approx(expected, abs=1e-12)

    # OTM under deterministic forward: put worth 0
    # S=100, K=90, T=1, r=0.05, sigma=0 -> K*exp(-r*T) = 85.6, < S=100 -> 0
    assert bsput(100.0, 90.0, 1.0, 0.05, 0.0) == 0.0


def test_zero_spot():
    """S = 0 -> guaranteed exercise, put worth K*exp(-r*T)."""
    price = bsput(0.0, 100.0, 1.0, 0.05, 0.20)
    expected = 100.0 * np.exp(-0.05)
    assert price == pytest.approx(expected, abs=1e-12)


def test_zero_strike():
    """K = 0 -> put worth nothing (no exercise ever profitable)."""
    assert bsput(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0


def test_nan_passthrough():
    """NaN in any input -> NaN out."""
    result = bsput(float("nan"), 100.0, 1.0, 0.05, 0.20)
    assert np.isnan(result)


def test_dtype_preserved_float32():
    """float32 inputs -> float32 output."""
    S = np.array([100.0], dtype=np.float32)
    K = np.array([100.0], dtype=np.float32)
    T = np.array([1.0], dtype=np.float32)
    r = np.array([0.05], dtype=np.float32)
    sigma = np.array([0.2], dtype=np.float32)

    result = bsput(S, K, T, r, sigma)
    assert result.dtype == np.float32
    # Looser tolerance for float32
    assert result[0] == pytest.approx(5.573526022256971, abs=1e-4)


def test_dtype_promoted_from_int():
    """Int inputs promoted to float64."""
    result = bsput(100, 100, 1, 1, 1)  # nonsense but shouldn't crash
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_price_monotonic_in_strike():
    """d(put)/dK > 0: higher strike -> higher put price, all else equal."""
    strikes = np.linspace(50, 150, 100)
    prices = bsput(100.0, strikes, 1.0, 0.05, 0.20)
    diffs = np.diff(prices)
    assert np.all(diffs > 0), "put must be strictly increasing in strike"


def test_price_monotonic_in_vol():
    """d(put)/d(sigma) > 0 (vega > 0): higher vol -> higher put price."""
    sigmas = np.linspace(0.05, 0.80, 50)
    prices = bsput(100.0, 100.0, 1.0, 0.05, sigmas)
    diffs = np.diff(prices)
    assert np.all(diffs > 0), "put must be strictly increasing in vol (positive vega)"


def test_price_nonnegative(rng):
    """Put price is always >= 0."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    prices = bsput(S, K, T, r, sigma)
    assert np.all(prices >= 0.0)


def test_price_bounded_by_strike(rng):
    """Put price <= K * exp(-r*T) (max possible payoff, discounted).

    A put can't be worth more than the discounted strike, because the
    maximum payoff at expiry is K (when S goes to 0).
    """
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    prices = bsput(S, K, T, r, sigma)
    upper_bound = K * np.exp(-r * T)
    assert np.all(prices <= upper_bound + 1e-12)


def test_put_call_parity(rng):
    """Put-Call Parity: C - P = S*exp(-q*T) - K*exp(-r*T).

    We don't have bscall yet, but we can back out the implied call from parity
    and check it's non-negative. Weak check but catches sign errors.
    """
    n = 100
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.1, 2.0, size=n)
    r = 0.05
    sigma = 0.25
    q = 0.02

    P = bsput(S, K, T, r, sigma, q)
    # C = P + S*exp(-q*T) - K*exp(-r*T)
    implied_C = P + S * np.exp(-q * T) - K * np.exp(-r * T)

    # Call price must be non-negative for a valid P
    assert np.all(implied_C >= -1e-12), "put-call parity gives negative call"


# ---------------------------------------------------------------------------
# 5. CPU == GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    """GPU-side bsput matches CPU-side bsput."""
    import cupy as cp

    n = 1000
    S_cpu = rng.uniform(50, 200, size=n)
    K_cpu = rng.uniform(50, 200, size=n)
    T_cpu = rng.uniform(0.01, 2.0, size=n)
    r_cpu = rng.uniform(0.0, 0.10, size=n)
    sigma_cpu = rng.uniform(0.05, 0.60, size=n)

    result_cpu = bsput(S_cpu, K_cpu, T_cpu, r_cpu, sigma_cpu)
    result_gpu = bsput(
        cp.asarray(S_cpu),
        cp.asarray(K_cpu),
        cp.asarray(T_cpu),
        cp.asarray(r_cpu),
        cp.asarray(sigma_cpu),
    )

    result_gpu_as_np = cp.asnumpy(result_gpu)
    np.testing.assert_allclose(result_cpu, result_gpu_as_np, atol=1e-10)


def test_gpu_preserves_backend(skip_no_gpu):
    """Any cupy input -> cupy output."""
    import cupy as cp

    result = bsput(cp.asarray([100.0]), 100.0, 1.0, 0.05, 0.20)
    assert isinstance(result, cp.ndarray)


def test_gpu_mixed_backend_promotes_to_gpu(skip_no_gpu):
    """Even one cupy input triggers full GPU computation."""
    import cupy as cp

    result = bsput(100.0, cp.asarray([90.0, 100.0, 110.0]), 1.0, 0.05, 0.20)
    assert isinstance(result, cp.ndarray)
    assert result.shape == (3,)

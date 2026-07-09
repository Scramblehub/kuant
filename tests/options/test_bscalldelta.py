"""Test suite for kuant.core.bscalldelta.

Includes the strongest cross-checks:
  - FD vs bscall
  - Put-call parity for delta: delta_call - delta_put = exp(-q*T)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bscall
from kuant.options import bscalldelta, bsputdelta


def _reference_delta(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, 0.6368306511756191),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0, 0.9286374026649281),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0, 0.28719163790512714),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03, 0.5621399977897841),
        (150.0, 100.0, 0.5, 0.05, 0.30, 0.0, 0.983631661789212),
        (50.0, 100.0, 0.5, 0.05, 0.30, 0.0, 0.0011687898309525743),
    ],
)
def test_golden_values(S, K, T, r, sigma, q, expected):
    result = bscalldelta(S, K, T, r, sigma, q)
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
        bscalldelta(S, K, T, r, sigma, q),
        _reference_delta(S, K, T, r, sigma, q),
        atol=1e-12,
        rtol=1e-12,
    )


def test_put_call_parity_for_delta(rng):
    """delta_call - delta_put = exp(-q*T). Machine-precision identity."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)

    dc = bscalldelta(S, K, T, r, sigma, q)
    dp = bsputdelta(S, K, T, r, sigma, q)
    np.testing.assert_allclose(dc - dp, np.exp(-q * T), atol=1e-13)


def test_delta_in_range(rng):
    """Call delta in [0, 1]."""
    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    q = rng.uniform(0.0, 0.05, size=n)
    deltas = bscalldelta(S, K, T, r, sigma, q)
    assert np.all(deltas >= 0.0 - 1e-15)
    assert np.all(deltas <= 1.0 + 1e-15)


def test_delta_monotonic_in_spot():
    """Higher S -> higher call delta (more ITM)."""
    spots = np.linspace(50, 150, 100)
    deltas = bscalldelta(spots, 100.0, 1.0, 0.05, 0.20)
    assert np.all(np.diff(deltas) > 0)


def test_delta_matches_finite_difference(rng):
    """delta ~ (bscall(S+h) - bscall(S-h)) / (2h)."""
    n = 20
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.2, 1.5, size=n)
    r, sigma = 0.05, 0.25
    h = 1e-4
    analytic = bscalldelta(S, K, T, r, sigma)
    fd = (bscall(S + h, K, T, r, sigma) - bscall(S - h, K, T, r, sigma)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, atol=1e-6)


# Edge cases
def test_expired_itm():
    assert bscalldelta(100.0, 80.0, 0.0, 0.05, 0.20) == 1.0


def test_expired_otm():
    assert bscalldelta(80.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_expired_atm():
    assert bscalldelta(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_zero_vol_exercise():
    """sigma=0, forward > K -> delta = exp(-q*T)."""
    # S=100, K=90, T=1, r=0.05, q=0.02: forward=100*e^-0.02=98.02, K*e^-0.05=85.6 -> exercise
    result = bscalldelta(100.0, 90.0, 1.0, 0.05, 0.0, 0.02)
    assert result == pytest.approx(np.exp(-0.02), abs=1e-12)


def test_zero_vol_no_exercise():
    """sigma=0, forward < K -> delta = 0."""
    # S=100, K=150, T=1, r=0.05, q=0: 100 < 150*e^-0.05=142.7 -> no exercise
    assert bscalldelta(100.0, 150.0, 1.0, 0.05, 0.0) == 0.0


def test_zero_spot():
    assert bscalldelta(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0


def test_zero_strike():
    """K=0 -> guaranteed exercise -> delta = exp(-q*T)."""
    assert bscalldelta(100.0, 0.0, 1.0, 0.05, 0.20, 0.03) == pytest.approx(np.exp(-0.03), abs=1e-12)


def test_nan_passthrough():
    assert np.isnan(bscalldelta(float("nan"), 100.0, 1.0, 0.05, 0.20))


def test_dtype_preserved_float32():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    result = bscalldelta(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    n = 500
    S = rng.uniform(50, 200, size=n)
    K = rng.uniform(50, 200, size=n)
    T = rng.uniform(0.01, 2.0, size=n)
    r = rng.uniform(0.0, 0.10, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    r_cpu = bscalldelta(S, K, T, r, sigma)
    r_gpu = bscalldelta(
        cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma)
    )
    np.testing.assert_allclose(r_cpu, cp.asnumpy(r_gpu), atol=1e-10)

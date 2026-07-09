"""Test suite for kuant.options.impvolbisection.

Validation strategy:
  1. Round-trip: bsput(sigma) -> impvolbisection -> sigma
  2. Match scipy.optimize.brentq (independent bisection)
  3. Match kuant.options.impvol (Newton) within tol
  4. No-arbitrage bounds -> NaN
  5. Edge cases: ATM, deep OTM/ITM, short tenor
  6. Both call and put
  7. Batched
  8. Convergence never diverges (unlike Newton)
  9. CPU==GPU parity
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from kuant.core import bscall, bsput
from kuant.options import impvol, impvolbisection


# ---------------------------------------------------------------------------
# 1. Round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sigma_true, S, K, T, r, q, is_call",
    [
        (0.20, 100.0, 100.0, 1.0, 0.05, 0.0, False),
        (0.20, 100.0, 100.0, 1.0, 0.05, 0.0, True),
        (0.15, 100.0, 110.0, 0.5, 0.03, 0.0, False),  # OTM put
        (0.35, 100.0, 90.0, 2.0, 0.05, 0.02, False),  # ITM put with div
        (0.10, 100.0, 90.0, 2.0, 0.05, 0.0, True),  # ITM call
        (0.60, 100.0, 100.0, 0.1, 0.05, 0.0, False),  # short tenor high vol
    ],
)
def test_round_trip(sigma_true, S, K, T, r, q, is_call):
    pricer = bscall if is_call else bsput
    price = pricer(S, K, T, r, sigma_true, q)
    sigma = impvolbisection(price, S, K, T, r, is_call=is_call, q=q)
    assert abs(sigma - sigma_true) < 1e-6


# ---------------------------------------------------------------------------
# 2. Match brentq reference
# ---------------------------------------------------------------------------


def _brentq_ref(price, S, K, T, r, is_call=False, q=0.0):
    pricer = bscall if is_call else bsput
    return brentq(
        lambda s: pricer(S, K, T, r, s, q) - price,
        1e-6,
        10.0,
        xtol=1e-8,
    )


def test_matches_brentq(rng):
    for _ in range(20):
        sigma_true = float(rng.uniform(0.05, 0.60))
        S = 100.0
        K = float(rng.uniform(60, 140))
        T = float(rng.uniform(0.1, 2.0))
        r = 0.05
        price = bsput(S, K, T, r, sigma_true)
        got = impvolbisection(price, S, K, T, r)
        ref = _brentq_ref(price, S, K, T, r)
        assert abs(got - ref) < 1e-6


# ---------------------------------------------------------------------------
# 3. Match kuant.options.impvol (Newton)
# ---------------------------------------------------------------------------


def test_matches_newton_impvol(rng):
    for _ in range(20):
        sigma_true = float(rng.uniform(0.05, 0.60))
        S = 100.0
        K = float(rng.uniform(70, 130))
        T = float(rng.uniform(0.2, 2.0))
        r = 0.05
        price = bsput(S, K, T, r, sigma_true)
        bis = impvolbisection(price, S, K, T, r)
        new = impvol(price, S, K, T, r)
        assert abs(bis - new) < 1e-5


# ---------------------------------------------------------------------------
# 4. No-arbitrage bounds
# ---------------------------------------------------------------------------


def test_price_below_intrinsic_nan():
    # Put intrinsic (K - S)+ at r=0 = max(K-S, 0) = 5
    # A quoted price of 1 (below intrinsic) has no solution
    result = impvolbisection(1.0, 100.0, 105.0, 1.0, 0.0)
    assert np.isnan(result)


def test_price_above_upper_nan():
    # Put upper bound = K·e^(-r·T); anything above has no solution
    result = impvolbisection(200.0, 100.0, 100.0, 1.0, 0.05)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


def test_ATM_put():
    price = bsput(100.0, 100.0, 1.0, 0.05, 0.20)
    assert abs(impvolbisection(price, 100.0, 100.0, 1.0, 0.05) - 0.20) < 1e-6


def test_high_vol():
    sigma_true = 2.0  # 200% annualized
    price = bsput(100.0, 100.0, 1.0, 0.05, sigma_true)
    assert abs(impvolbisection(price, 100.0, 100.0, 1.0, 0.05) - sigma_true) < 1e-5


def test_very_short_tenor():
    price = bsput(100.0, 100.0, 0.01, 0.05, 0.30)
    got = impvolbisection(price, 100.0, 100.0, 0.01, 0.05)
    assert abs(got - 0.30) < 1e-5


# ---------------------------------------------------------------------------
# 6. Batched
# ---------------------------------------------------------------------------


def test_batched_matches_scalar():
    sigmas = np.array([0.15, 0.20, 0.25, 0.30])
    prices = np.array([bsput(100.0, 100.0, 1.0, 0.05, s) for s in sigmas])
    got = impvolbisection(prices, 100.0, 100.0, 1.0, 0.05)
    np.testing.assert_allclose(got, sigmas, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Never diverges — bracket always shrinks
# ---------------------------------------------------------------------------


def test_bracket_shrinks_monotonically(rng):
    """Bisection guarantees hi-lo strictly halves each iteration until convergence.
    Verified indirectly: 30 iterations is more than enough for tol=1e-8 with a
    bracket of width 10."""
    for _ in range(10):
        sigma_true = float(rng.uniform(0.1, 0.5))
        price = bsput(100.0, 100.0, 1.0, 0.05, sigma_true)
        got = impvolbisection(price, 100.0, 100.0, 1.0, 0.05, max_iter=30)
        assert abs(got - sigma_true) < 1e-6


# ---------------------------------------------------------------------------
# 8. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    sigmas = rng.uniform(0.1, 0.5, 20)
    S = np.full(20, 100.0)
    K = rng.uniform(80, 120, 20)
    T = rng.uniform(0.2, 2.0, 20)
    r = np.full(20, 0.05)
    prices = np.array([bsput(100.0, K[i], T[i], 0.05, sigmas[i]) for i in range(20)])
    r_cpu = impvolbisection(prices, S, K, T, r)
    r_gpu = cp.asnumpy(
        impvolbisection(
            cp.asarray(prices), cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r)
        )
    )
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-6)

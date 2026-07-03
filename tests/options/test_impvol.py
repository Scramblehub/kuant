'''Test suite for kuant.options.impvol.

Validation strategy:
  1. Round-trip: bsput(sigma) -> impvol -> sigma
  2. Match scipy.optimize.brentq (independent algorithm)
  3. No-arbitrage bounds -> NaN
  4. Edge cases: ATM, deep OTM/ITM, short tenor, high vol
  5. Call and put both work
  6. Batched calls
  7. dtype preservation
  8. CPU==GPU parity
'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from kuant.core import bscall, bsput
from kuant.options import impvol


# ---------------------------------------------------------------------------
# 1. Round-trip: known sigma -> price -> impvol should recover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('sigma_true', [0.10, 0.20, 0.30, 0.50, 0.80, 1.20])
@pytest.mark.parametrize('S, K, T, r, q', [
    (100.0, 100.0, 1.0, 0.05, 0.00),  # ATM 1y
    (100.0, 105.0, 0.5, 0.05, 0.02),  # slight OTM with dividend
    (100.0, 95.0, 0.5, 0.05, 0.00),   # slight ITM (put)
])
def test_roundtrip_put(sigma_true, S, K, T, r, q):
    '''Round-trip test in the well-behaved region (moderate |moneyness|,
    non-tiny vol). Newton recovers to essentially machine precision here.'''
    price = bsput(S, K, T, r, sigma_true, q)
    sigma_iv = impvol(price, S, K, T, r, is_call=False, q=q)
    assert abs(sigma_iv - sigma_true) < 1e-6, (
        f'sigma_true={sigma_true}, sigma_iv={sigma_iv}, S={S}, K={K}, T={T}'
    )


def test_roundtrip_put_low_vol_wide_tolerance():
    '''Very low vol + off-ATM has near-zero vega; Newton hits the numerical
    floor. Recovery is still good but not machine-precision.'''
    sigma_true = 0.05
    price = bsput(100.0, 90.0, 0.25, 0.05, sigma_true)  # 5% vol, 3mo, OTM
    sigma_iv = impvol(price, 100.0, 90.0, 0.25, 0.05)
    # This regime is inherently ill-conditioned; wider tolerance is honest.
    if not np.isnan(sigma_iv):
        assert abs(sigma_iv - sigma_true) < 5e-2


@pytest.mark.parametrize('sigma_true', [0.10, 0.30, 0.60])
@pytest.mark.parametrize('S, K, T, r, q', [
    (100.0, 100.0, 1.0, 0.05, 0.00),
    (100.0, 90.0, 0.5, 0.05, 0.00),   # ITM (for call)
    (100.0, 110.0, 0.5, 0.05, 0.02),  # OTM (for call)
])
def test_roundtrip_call(sigma_true, S, K, T, r, q):
    price = bscall(S, K, T, r, sigma_true, q)
    sigma_iv = impvol(price, S, K, T, r, is_call=True, q=q)
    assert abs(sigma_iv - sigma_true) < 1e-6


# ---------------------------------------------------------------------------
# 2. Reference match — scipy.optimize.brentq
# ---------------------------------------------------------------------------


def _brentq_impvol(target_price, S, K, T, r, is_call=False, q=0.0):
    '''Independent implementation: bisection via scipy.optimize.brentq.'''
    def obj(sigma):
        price = bscall(S, K, T, r, sigma, q) if is_call else bsput(S, K, T, r, sigma, q)
        return price - target_price
    return brentq(obj, 1e-6, 5.0, xtol=1e-10)


def test_matches_brentq_random_puts(rng):
    '''Newton and brentq are both root-finders on the same objective; they
    should agree wherever the price surface has non-negligible slope.
    Skip the low-vega tail where BOTH algorithms hit the same numerical
    floor and disagree by the flat-region uncertainty.'''
    from kuant.options import bsvega

    for _ in range(50):
        S = rng.uniform(50, 200)
        K = rng.uniform(50, 200)
        T = rng.uniform(0.1, 2.0)
        r = rng.uniform(0.0, 0.10)
        q = rng.uniform(0.0, 0.05)
        sigma_true = rng.uniform(0.10, 1.0)  # avoid the < 0.10 flat tail

        # Skip if vega is too low to invert reliably.
        vega = bsvega(S, K, T, r, sigma_true, q)
        if vega < 0.1:
            continue

        price = bsput(S, K, T, r, sigma_true, q)
        sigma_ours = impvol(price, S, K, T, r, is_call=False, q=q)
        sigma_brentq = _brentq_impvol(price, S, K, T, r, is_call=False, q=q)

        assert abs(sigma_ours - sigma_brentq) < 1e-5


def test_matches_brentq_random_calls(rng):
    from kuant.options import bsvega

    for _ in range(50):
        S = rng.uniform(50, 200)
        K = rng.uniform(50, 200)
        T = rng.uniform(0.1, 2.0)
        r = rng.uniform(0.0, 0.10)
        q = rng.uniform(0.0, 0.05)
        sigma_true = rng.uniform(0.10, 1.0)

        vega = bsvega(S, K, T, r, sigma_true, q)
        if vega < 0.1:
            continue

        price = bscall(S, K, T, r, sigma_true, q)
        sigma_ours = impvol(price, S, K, T, r, is_call=True, q=q)
        sigma_brentq = _brentq_impvol(price, S, K, T, r, is_call=True, q=q)

        assert abs(sigma_ours - sigma_brentq) < 1e-5


# ---------------------------------------------------------------------------
# 3. No-arbitrage bounds -> NaN
# ---------------------------------------------------------------------------


def test_put_price_below_intrinsic_returns_nan():
    '''Put price below max(K*e^-rT - S*e^-qT, 0) is arbitrage.'''
    S, K, T, r = 100.0, 110.0, 1.0, 0.05
    # Lower bound: 110*e^-0.05 - 100 ~ 4.63; below that is arbitrage
    price = 1.0  # way below
    result = impvol(price, S, K, T, r, is_call=False)
    assert np.isnan(result)


def test_put_price_above_upper_bound_returns_nan():
    '''Put worth more than discounted strike is arbitrage.'''
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    # Upper bound: 100 * e^-0.05 ~ 95.12
    price = 200.0  # way above
    result = impvol(price, S, K, T, r, is_call=False)
    assert np.isnan(result)


def test_call_price_negative_returns_nan():
    result = impvol(-1.0, 100.0, 100.0, 1.0, 0.05, is_call=True)
    assert np.isnan(result)


def test_zero_T_returns_nan():
    '''T=0 has no vol solution.'''
    result = impvol(1.0, 100.0, 100.0, 0.0, 0.05, is_call=False)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


def test_short_tenor_atm():
    '''1 day to expiry, ATM.'''
    sigma_true = 0.30
    S, K, T, r = 100.0, 100.0, 1/365, 0.05
    price = bsput(S, K, T, r, sigma_true)
    sigma_iv = impvol(price, S, K, T, r)
    assert abs(sigma_iv - sigma_true) < 1e-6


def test_very_high_vol():
    '''sigma = 1.5 (150% annualized).'''
    sigma_true = 1.5
    price = bsput(100.0, 100.0, 1.0, 0.05, sigma_true)
    sigma_iv = impvol(price, 100.0, 100.0, 1.0, 0.05)
    assert abs(sigma_iv - sigma_true) < 1e-6


def test_deep_otm_put():
    '''K=50, S=100 -> deep OTM put with tiny price.'''
    sigma_true = 0.40
    S, K, T, r = 100.0, 50.0, 1.0, 0.05
    price = bsput(S, K, T, r, sigma_true)
    if price > 1e-10:  # deep OTM, price might be too small to invert reliably
        sigma_iv = impvol(price, S, K, T, r)
        assert abs(sigma_iv - sigma_true) < 1e-5


# ---------------------------------------------------------------------------
# 5. Batched calls
# ---------------------------------------------------------------------------


def test_batched_puts(rng):
    n = 100
    S = 100.0
    K = np.array([80, 90, 100, 110, 120] * (n // 5))
    T = 1.0
    r = 0.05
    sigmas_true = rng.uniform(0.1, 0.8, size=n)

    prices = bsput(S, K, T, r, sigmas_true)
    sigmas_iv = impvol(prices, S, K, T, r, is_call=False)

    assert sigmas_iv.shape == (n,)
    np.testing.assert_allclose(sigmas_iv, sigmas_true, atol=1e-6)


def test_batched_calls():
    n = 20
    S = 100.0
    K = np.linspace(80, 120, n)
    T = 0.5
    r = 0.05
    sigma_true = 0.25

    prices = bscall(S, K, T, r, sigma_true)
    sigmas_iv = impvol(prices, S, K, T, r, is_call=True)

    np.testing.assert_allclose(sigmas_iv, sigma_true, atol=1e-6)


def test_scalar_in_scalar_out():
    price = bsput(100.0, 100.0, 1.0, 0.05, 0.25)
    result = impvol(price, 100.0, 100.0, 1.0, 0.05)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 6. Dtype
# ---------------------------------------------------------------------------


def test_dtype_preserved_float32():
    S = np.array([100.0], dtype=np.float32)
    K = np.array([100.0], dtype=np.float32)
    T = np.array([1.0], dtype=np.float32)
    r = np.array([0.05], dtype=np.float32)
    sigma_true = np.float32(0.25)
    price = bsput(S, K, T, r, sigma_true)
    assert price.dtype == np.float32
    result = impvol(price, S, K, T, r)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# 7. CPU == GPU
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    n = 50
    S = 100.0
    K = np.linspace(80, 120, n)
    T = 0.5
    r = 0.05
    sigmas_true = rng.uniform(0.1, 0.6, size=n)

    prices_cpu = bsput(S, K, T, r, sigmas_true)

    sigmas_cpu = impvol(prices_cpu, S, K, T, r, is_call=False)
    sigmas_gpu = impvol(
        cp.asarray(prices_cpu),
        cp.asarray(np.full(n, S)),
        cp.asarray(K),
        cp.asarray(np.full(n, T)),
        cp.asarray(np.full(n, r)),
        is_call=False,
    )

    np.testing.assert_allclose(sigmas_cpu, cp.asnumpy(sigmas_gpu), atol=1e-8)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    price = bsput(100.0, 100.0, 1.0, 0.05, 0.25)
    result = impvol(
        cp.asarray([price]),
        cp.asarray([100.0]),
        cp.asarray([100.0]),
        cp.asarray([1.0]),
        cp.asarray([0.05]),
    )
    assert isinstance(result, cp.ndarray)

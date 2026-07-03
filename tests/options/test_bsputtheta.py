'''Test suite for kuant.options.bsputtheta.

Validation strategy:
  1. Match scipy analytic reference
  2. Match finite-difference of bsput
  3. Put-call parity for theta (cross-check in bscalltheta test)
  4. Edge cases: expired, S=0, K=0, deep OTM/ITM
  5. Batched input
  6. dtype preservation
  7. CPU==GPU parity
'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import bsput
from kuant.options import bsputtheta


def _ref_put_theta(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (-S*np.exp(-q*T)*norm.pdf(d1)*sigma / (2*np.sqrt(T))
            + r*K*np.exp(-r*T)*norm.cdf(-d2)
            - q*S*np.exp(-q*T)*norm.cdf(-d1))


@pytest.mark.parametrize(
    'S, K, T, r, sigma, q',
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0),   # ITM put
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0),    # OTM put
        (100.0, 100.0, 0.25, 0.05, 0.20, 0.0),
        (100.0, 100.0, 2.0, 0.05, 0.30, 0.0),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03),
        (100.0, 100.0, 1.0, 0.00, 0.20, 0.00),
    ],
)
def test_matches_scipy_reference(S, K, T, r, sigma, q):
    assert abs(bsputtheta(S, K, T, r, sigma, q) - _ref_put_theta(S, K, T, r, sigma, q)) < 1e-10


def test_finite_difference_matches():
    '''theta = -d(price)/dT. h=1e-5 balances truncation and roundoff.'''
    S, K, r, sigma, q = 100.0, 100.0, 0.05, 0.20, 0.02
    for T in [0.25, 0.5, 1.0, 2.0]:
        dt = 1e-5
        num = -(bsput(S, K, T + dt, r, sigma, q) - bsput(S, K, T - dt, r, sigma, q)) / (2 * dt)
        ana = bsputtheta(S, K, T, r, sigma, q)
        assert abs(ana - num) < 1e-8


def test_expired_zero():
    assert bsputtheta(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_K_zero_gives_zero():
    '''Put with K=0 is worthless -> theta 0.'''
    assert bsputtheta(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0


def test_S_zero_gives_rK_growth():
    '''When S=0 the put is worth K·e^(-r·T); theta = r·K·e^(-r·T).'''
    K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
    expected = r * K * np.exp(-r * T)
    assert abs(bsputtheta(0.0, K, T, r, sigma) - expected) < 1e-10


def test_ATM_put_theta_negative():
    assert bsputtheta(100.0, 100.0, 1.0, 0.05, 0.20) < 0


def test_deep_ITM_put_can_be_positive():
    '''Deep-ITM European put with high rates can have positive theta.'''
    result = bsputtheta(60.0, 100.0, 1.0, 0.10, 0.20)
    assert result > 0


def test_batched_matches_scalar():
    S_arr = np.array([80.0, 100.0, 120.0])
    result = bsputtheta(S_arr, 100.0, 1.0, 0.05, 0.20)
    for i, S in enumerate([80.0, 100.0, 120.0]):
        assert abs(result[i] - bsputtheta(S, 100.0, 1.0, 0.05, 0.20)) < 1e-12


def test_dtype_float32_preserved():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    result = bsputtheta(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    S = rng.uniform(50, 150, 100)
    K = rng.uniform(50, 150, 100)
    T = rng.uniform(0.1, 2.0, 100)
    r = rng.uniform(-0.01, 0.10, 100)
    sigma = rng.uniform(0.1, 0.6, 100)
    r_cpu = bsputtheta(S, K, T, r, sigma)
    r_gpu = cp.asnumpy(bsputtheta(cp.asarray(S), cp.asarray(K), cp.asarray(T),
                                   cp.asarray(r), cp.asarray(sigma)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

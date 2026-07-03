'''Test suite for kuant.options.bscallcharm.

Sign convention: charm = -dDelta/dT (delta bleed per year).

Validation strategy:
  1. Match scipy analytic reference
  2. Match -1 * finite-difference of bscalldelta w.r.t. T
  3. Put-call charm parity
  4. Edge cases: expired, S=0, K=0, deep OTM/ITM
  5. Batched input
  6. dtype preservation
  7. CPU==GPU parity
'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.options import bscallcharm, bscalldelta, bsputcharm


def _ref_call_charm(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    inner = (2*(r-q)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
    return q*np.exp(-q*T)*norm.cdf(d1) - np.exp(-q*T)*norm.pdf(d1)*inner


@pytest.mark.parametrize(
    'S, K, T, r, sigma, q',
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 100.0, 0.25, 0.05, 0.20, 0.0),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03),
        (100.0, 100.0, 1.0, 0.00, 0.20, 0.00),
    ],
)
def test_matches_scipy_reference(S, K, T, r, sigma, q):
    assert abs(bscallcharm(S, K, T, r, sigma, q) - _ref_call_charm(S, K, T, r, sigma, q)) < 1e-10


def test_finite_difference_matches():
    '''charm = -d(delta)/dT.'''
    S, K, r, sigma, q = 100.0, 100.0, 0.05, 0.20, 0.02
    for T in [0.25, 0.5, 1.0, 2.0]:
        dt = 1e-5
        num_ddelta_dT = (bscalldelta(S, K, T + dt, r, sigma, q)
                         - bscalldelta(S, K, T - dt, r, sigma, q)) / (2 * dt)
        ana_charm = bscallcharm(S, K, T, r, sigma, q)
        assert abs(ana_charm - (-num_ddelta_dT)) < 1e-5


def test_call_put_charm_parity():
    '''From delta parity (delta_c - delta_p = e^(-qT)) differentiated w.r.t. T:
       d(delta_c)/dT - d(delta_p)/dT = -q·e^(-qT)
       charm_c - charm_p = -(-q·e^(-qT)) = +q·e^(-qT)
    '''
    for S, K, T, r, sigma, q in [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.02),
        (100.0, 120.0, 0.5, 0.03, 0.15, 0.01),
    ]:
        lhs = bscallcharm(S, K, T, r, sigma, q) - bsputcharm(S, K, T, r, sigma, q)
        rhs = q * np.exp(-q * T)
        assert abs(lhs - rhs) < 1e-10


def test_expired_zero():
    assert bscallcharm(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_S_zero_gives_zero():
    assert bscallcharm(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0


def test_K_zero_gives_neg_q_disc():
    '''K=0 -> delta = e^(-q·T), so charm = -d(delta)/dT = -q·e^(-q·T).'''
    S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.03
    expected = -q * np.exp(-q * T)
    assert abs(bscallcharm(S, 0.0, T, r, sigma, q) - expected) < 1e-10


def test_zero_vol_zero_away_from_jump():
    result = bscallcharm(100.0, 200.0, 1.0, 0.0, 0.0)
    assert result == 0.0


def test_batched_matches_scalar():
    S_arr = np.array([90.0, 100.0, 110.0])
    result = bscallcharm(S_arr, 100.0, 1.0, 0.05, 0.20)
    for i, S in enumerate([90.0, 100.0, 110.0]):
        assert abs(result[i] - bscallcharm(S, 100.0, 1.0, 0.05, 0.20)) < 1e-12


def test_dtype_float32_preserved():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    result = bscallcharm(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    S = rng.uniform(50, 150, 100)
    K = rng.uniform(50, 150, 100)
    T = rng.uniform(0.1, 2.0, 100)
    r = rng.uniform(-0.01, 0.10, 100)
    sigma = rng.uniform(0.1, 0.6, 100)
    r_cpu = bscallcharm(S, K, T, r, sigma)
    r_gpu = cp.asnumpy(bscallcharm(cp.asarray(S), cp.asarray(K), cp.asarray(T),
                                    cp.asarray(r), cp.asarray(sigma)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

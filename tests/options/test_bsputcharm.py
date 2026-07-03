'''Test suite for kuant.options.bsputcharm.

Sign convention: charm = -dDelta_put/dT.
'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.options import bsputcharm, bsputdelta


def _ref_put_charm(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    inner = (2*(r-q)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
    return -q*np.exp(-q*T)*norm.cdf(-d1) - np.exp(-q*T)*norm.pdf(d1)*inner


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
    assert abs(bsputcharm(S, K, T, r, sigma, q) - _ref_put_charm(S, K, T, r, sigma, q)) < 1e-10


def test_finite_difference_matches():
    '''charm = -d(delta)/dT. h=1e-5 balances truncation and roundoff.'''
    S, K, r, sigma, q = 100.0, 100.0, 0.05, 0.20, 0.02
    for T in [0.25, 0.5, 1.0, 2.0]:
        dt = 1e-5
        num_ddelta_dT = (bsputdelta(S, K, T + dt, r, sigma, q)
                         - bsputdelta(S, K, T - dt, r, sigma, q)) / (2 * dt)
        ana_charm = bsputcharm(S, K, T, r, sigma, q)
        assert abs(ana_charm - (-num_ddelta_dT)) < 1e-9


def test_expired_zero():
    assert bsputcharm(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_K_zero_gives_zero():
    assert bsputcharm(100.0, 0.0, 1.0, 0.05, 0.20) == 0.0


def test_S_zero_gives_neg_q_disc():
    '''At S=0 put_delta = -e^(-q·T), so charm = -d(delta)/dT = -q·e^(-q·T).'''
    K, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.03
    expected = -q * np.exp(-q * T)
    assert abs(bsputcharm(0.0, K, T, r, sigma, q) - expected) < 1e-10


def test_batched_matches_scalar():
    S_arr = np.array([80.0, 100.0, 120.0])
    result = bsputcharm(S_arr, 100.0, 1.0, 0.05, 0.20)
    for i, S in enumerate([80.0, 100.0, 120.0]):
        assert abs(result[i] - bsputcharm(S, 100.0, 1.0, 0.05, 0.20)) < 1e-12


def test_dtype_float32_preserved():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    result = bsputcharm(*args)
    assert result.dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    S = rng.uniform(50, 150, 100)
    K = rng.uniform(50, 150, 100)
    T = rng.uniform(0.1, 2.0, 100)
    r = rng.uniform(-0.01, 0.10, 100)
    sigma = rng.uniform(0.1, 0.6, 100)
    r_cpu = bsputcharm(S, K, T, r, sigma)
    r_gpu = cp.asnumpy(bsputcharm(cp.asarray(S), cp.asarray(K), cp.asarray(T),
                                   cp.asarray(r), cp.asarray(sigma)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

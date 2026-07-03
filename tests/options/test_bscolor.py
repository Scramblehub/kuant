'''Test suite for kuant.options.bscolor.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.options import bscolor, bsgamma


def _ref_color(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    inner = 2*q*T + 1 + (2*(r-q)*T - d2*sigma*np.sqrt(T))*d1/(sigma*np.sqrt(T))
    prefactor = -np.exp(-q*T)*norm.pdf(d1)/(2*S*T*sigma*np.sqrt(T))
    return prefactor * inner


@pytest.mark.parametrize(
    'S, K, T, r, sigma, q',
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 100.0, 0.25, 0.05, 0.20, 0.0),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03),
    ],
)
def test_matches_scipy_reference(S, K, T, r, sigma, q):
    assert abs(bscolor(S, K, T, r, sigma, q) - _ref_color(S, K, T, r, sigma, q)) < 1e-10


def test_fd_dGamma_dT():
    '''color = d(gamma)/d(T). h=1e-5 balances truncation and roundoff.'''
    S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.05, 0.20, 0.02
    dt = 1e-5
    fd = (bsgamma(S, K, T + dt, r, sigma, q) - bsgamma(S, K, T - dt, r, sigma, q)) / (2 * dt)
    assert abs(bscolor(S, K, T, r, sigma, q) - fd) < 1e-9


def test_expired_zero():
    assert bscolor(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_S_zero_zero():
    assert bscolor(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0


def test_batched():
    S = np.array([80.0, 100.0, 120.0])
    result = bscolor(S, 100.0, 1.0, 0.05, 0.20)
    for i, s in enumerate([80.0, 100.0, 120.0]):
        assert abs(result[i] - bscolor(s, 100.0, 1.0, 0.05, 0.20)) < 1e-12


def test_dtype_float32_preserved():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    assert bscolor(*args).dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    S = rng.uniform(50, 150, 100)
    K = rng.uniform(50, 150, 100)
    T = rng.uniform(0.1, 2.0, 100)
    r = rng.uniform(-0.01, 0.10, 100)
    sigma = rng.uniform(0.1, 0.6, 100)
    r_cpu = bscolor(S, K, T, r, sigma)
    r_gpu = cp.asnumpy(bscolor(cp.asarray(S), cp.asarray(K), cp.asarray(T),
                                cp.asarray(r), cp.asarray(sigma)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

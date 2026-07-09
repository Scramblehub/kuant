"""Test suite for kuant.options.bsvanna.

Validation strategy:
  1. Match scipy analytic reference
  2. Match finite-difference of bscalldelta w.r.t. sigma
  3. Match finite-difference of bsvega w.r.t. spot
  4. Put-call symmetry: vanna_call == vanna_put
  5. Edge cases
  6. Batched
  7. dtype preservation
  8. CPU==GPU parity
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.options import bscalldelta, bsputdelta, bsvanna, bsvega


def _ref_vanna(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 120.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 80.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 100.0, 0.25, 0.05, 0.20, 0.0),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.03),
    ],
)
def test_matches_scipy_reference(S, K, T, r, sigma, q):
    assert abs(bsvanna(S, K, T, r, sigma, q) - _ref_vanna(S, K, T, r, sigma, q)) < 1e-10


def test_fd_dDelta_dSigma():
    """vanna = d(delta_call)/d(sigma). Step chosen for O(1e-11) truncation."""
    S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.05, 0.20, 0.02
    ds = 1e-6
    fd = (bscalldelta(S, K, T, r, sigma + ds, q) - bscalldelta(S, K, T, r, sigma - ds, q)) / (
        2 * ds
    )
    assert abs(bsvanna(S, K, T, r, sigma, q) - fd) < 1e-9


def test_fd_dVega_dSpot():
    """vanna = d(vega)/d(spot). h=1e-4 balances truncation and roundoff."""
    S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.05, 0.20, 0.02
    ds = 1e-4
    fd = (bsvega(S + ds, K, T, r, sigma, q) - bsvega(S - ds, K, T, r, sigma, q)) / (2 * ds)
    assert abs(bsvanna(S, K, T, r, sigma, q) - fd) < 1e-9


def test_put_call_symmetry():
    """Vanna is put-call symmetric: dp/dsigma = dc/dsigma - d(e^(-qT))/dsigma = dc/dsigma."""
    S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.05, 0.20, 0.02
    ds = 1e-6
    vanna_call = (
        bscalldelta(S, K, T, r, sigma + ds, q) - bscalldelta(S, K, T, r, sigma - ds, q)
    ) / (2 * ds)
    vanna_put = (bsputdelta(S, K, T, r, sigma + ds, q) - bsputdelta(S, K, T, r, sigma - ds, q)) / (
        2 * ds
    )
    assert abs(vanna_call - vanna_put) < 1e-6


def test_expired_zero():
    assert bsvanna(100.0, 100.0, 0.0, 0.05, 0.20) == 0.0


def test_S_zero_gives_zero():
    assert bsvanna(0.0, 100.0, 1.0, 0.05, 0.20) == 0.0


def test_zero_vol_zero():
    assert bsvanna(100.0, 100.0, 1.0, 0.05, 0.0) == 0.0


def test_batched():
    S = np.array([80.0, 100.0, 120.0])
    result = bsvanna(S, 100.0, 1.0, 0.05, 0.20)
    for i, s in enumerate([80.0, 100.0, 120.0]):
        assert abs(result[i] - bsvanna(s, 100.0, 1.0, 0.05, 0.20)) < 1e-12


def test_dtype_float32_preserved():
    args = [np.array([100.0], dtype=np.float32)] * 2 + [
        np.array([1.0], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    ]
    assert bsvanna(*args).dtype == np.float32


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    S = rng.uniform(50, 150, 100)
    K = rng.uniform(50, 150, 100)
    T = rng.uniform(0.1, 2.0, 100)
    r = rng.uniform(-0.01, 0.10, 100)
    sigma = rng.uniform(0.1, 0.6, 100)
    r_cpu = bsvanna(S, K, T, r, sigma)
    r_gpu = cp.asnumpy(
        bsvanna(cp.asarray(S), cp.asarray(K), cp.asarray(T), cp.asarray(r), cp.asarray(sigma))
    )
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

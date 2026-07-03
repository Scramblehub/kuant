"""Benchmarks for kuant.options — Greeks, payoffs, impvol solvers, chain filters."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.core import bsput
from kuant.options import (
    bscalldelta,
    bscallcharm,
    bscalltheta,
    bsgamma,
    bsvanna,
    bsvega,
    callpayoff,
    deltabucket,
    impvol,
    impvolbisection,
    moneynessbucket,
)

try:
    import cupy as cp

    _HAS_GPU = cp.cuda.is_available()
except (ImportError, RuntimeError):
    cp = None
    _HAS_GPU = False


# ---------- First-order Greeks — batched 1K ---------------------------------


def _grid_1k(rng):
    K = rng.uniform(80, 120, 1_000)
    return K


def test_bench_bscalldelta_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = _grid_1k(rng)
    benchmark(bscalldelta, 100.0, K, 1.0, 0.05, 0.20)


def test_bench_bsgamma_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = _grid_1k(rng)
    benchmark(bsgamma, 100.0, K, 1.0, 0.05, 0.20)


def test_bench_bsvega_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = _grid_1k(rng)
    benchmark(bsvega, 100.0, K, 1.0, 0.05, 0.20)


def test_bench_bscalltheta_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = _grid_1k(rng)
    benchmark(bscalltheta, 100.0, K, 1.0, 0.05, 0.20)


# ---------- Second-order Greeks — batched 1K --------------------------------


def test_bench_bsvanna_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = _grid_1k(rng)
    benchmark(bsvanna, 100.0, K, 1.0, 0.05, 0.20)


def test_bench_bscallcharm_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = _grid_1k(rng)
    benchmark(bscallcharm, 100.0, K, 1.0, 0.05, 0.20)


# ---------- Payoffs — trivial but often hot ---------------------------------


def test_bench_callpayoff_batch_1m(benchmark, rng=np.random.default_rng(0)):
    S = rng.uniform(80, 120, 1_000_000)
    benchmark(callpayoff, S, 100.0)


# ---------- Chain filters ---------------------------------------------------


def test_bench_deltabucket_100_targets(benchmark, rng=np.random.default_rng(0)):
    chain_deltas = np.sort(rng.uniform(0, 1, 500))
    targets = rng.uniform(0, 1, 100)
    benchmark(deltabucket, chain_deltas, targets)


def test_bench_moneynessbucket_batch_1k(benchmark, rng=np.random.default_rng(0)):
    S = np.full(1_000, 100.0)
    K = rng.uniform(60, 140, 1_000)
    T = rng.uniform(0.1, 2.0, 1_000)
    benchmark(moneynessbucket, S, K, T, 0.05, 0.02)


# ---------- IV solvers — the marquee comparison -----------------------------


def _synthetic_prices(rng, n=100):
    K = np.linspace(85, 115, n)
    sigma_true = 0.20 + 0.1 * np.log(K / 100.0) ** 2  # skew
    price = bsput(100.0, K, 1.0, 0.05, sigma_true)
    return price, K


def test_bench_impvol_newton_100(benchmark, rng=np.random.default_rng(0)):
    price, K = _synthetic_prices(rng)
    benchmark(impvol, price, 100.0, K, 1.0, 0.05, is_call=False)


def test_bench_impvol_bisection_100(benchmark, rng=np.random.default_rng(0)):
    price, K = _synthetic_prices(rng)
    benchmark(impvolbisection, price, 100.0, K, 1.0, 0.05, is_call=False)


# ---------- GPU ------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA not available")
def test_bench_bscalldelta_batch_1m_gpu(benchmark, rng=np.random.default_rng(0)):
    K = cp.asarray(rng.uniform(80, 120, 1_000_000))
    benchmark(bscalldelta, 100.0, K, 1.0, 0.05, 0.20)


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA not available")
def test_bench_bsgamma_batch_1m_gpu(benchmark, rng=np.random.default_rng(0)):
    K = cp.asarray(rng.uniform(80, 120, 1_000_000))
    benchmark(bsgamma, 100.0, K, 1.0, 0.05, 0.20)

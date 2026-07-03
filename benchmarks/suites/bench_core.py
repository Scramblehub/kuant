"""Benchmarks for kuant.core primitives.

Covers the hot paths users care about:
- BS pricing (scalar + batched)
- Gaussian CDF/PDF (scalar + batched)
- log-space primitives (logsumexp, lognormcdf)
- Fat-tail primitives (tcdf, gpdcdf) — routed via _special_bridge

Each benchmark ships in two variants where applicable:
- _cpu   — numpy input
- _gpu   — cupy input (skipped without CUDA)

The queue runner normalizes rounds and captures min/median/mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from kuant.core import (
    bscall,
    bsput,
    gpdcdf,
    lognormcdf,
    logsumexp,
    normcdf,
    normpdf,
    normppf,
    tcdf,
)

# Try to import cupy; GPU benchmarks skip gracefully if unavailable.
try:
    import cupy as cp

    _HAS_GPU = cp.cuda.is_available()
except (ImportError, RuntimeError):
    cp = None
    _HAS_GPU = False


# ---------- CPU: scalar (dispatch-overhead-bound) ---------------------------


def test_bench_bscall_scalar(benchmark):
    benchmark(bscall, 100.0, 100.0, 1.0, 0.05, 0.20)


def test_bench_normcdf_scalar(benchmark):
    benchmark(normcdf, 0.5)


def test_bench_normppf_scalar(benchmark):
    benchmark(normppf, 0.975)


# ---------- CPU: batched 1K (mixed compute + dispatch) ----------------------


def test_bench_bscall_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = rng.uniform(80, 120, 1_000)
    benchmark(bscall, 100.0, K, 1.0, 0.05, 0.20)


def test_bench_bsput_batch_1k(benchmark, rng=np.random.default_rng(0)):
    K = rng.uniform(80, 120, 1_000)
    benchmark(bsput, 100.0, K, 1.0, 0.05, 0.20)


def test_bench_normcdf_batch_1k(benchmark, rng=np.random.default_rng(0)):
    x = rng.uniform(-3, 3, 1_000)
    benchmark(normcdf, x)


def test_bench_normpdf_batch_1k(benchmark, rng=np.random.default_rng(0)):
    x = rng.uniform(-3, 3, 1_000)
    benchmark(normpdf, x)


def test_bench_logsumexp_batch_1k(benchmark, rng=np.random.default_rng(0)):
    x = rng.uniform(-100, 100, 1_000)
    benchmark(logsumexp, x)


def test_bench_lognormcdf_batch_1k(benchmark, rng=np.random.default_rng(0)):
    x = rng.uniform(-5, 5, 1_000)
    benchmark(lognormcdf, x)


# ---------- CPU: batched 1M (compute-bound) ---------------------------------


def test_bench_bscall_batch_1m(benchmark, rng=np.random.default_rng(0)):
    K = rng.uniform(80, 120, 1_000_000)
    benchmark(bscall, 100.0, K, 1.0, 0.05, 0.20)


def test_bench_normcdf_batch_1m(benchmark, rng=np.random.default_rng(0)):
    x = rng.uniform(-3, 3, 1_000_000)
    benchmark(normcdf, x)


# ---------- Fat-tail special functions (bridge-routed) ----------------------


def test_bench_tcdf_batch_1k(benchmark, rng=np.random.default_rng(0)):
    x = rng.uniform(-3, 3, 1_000)
    benchmark(tcdf, x, 5.0)


def test_bench_gpdcdf_batch_1k(benchmark, rng=np.random.default_rng(0)):
    x = rng.uniform(0.001, 3, 1_000)
    benchmark(gpdcdf, x, 0.3, 1.0)


# ---------- GPU: batched 1M ------------------------------------------------


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA not available")
def test_bench_bscall_batch_1m_gpu(benchmark, rng=np.random.default_rng(0)):
    K = cp.asarray(rng.uniform(80, 120, 1_000_000))
    benchmark(bscall, 100.0, K, 1.0, 0.05, 0.20)


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA not available")
def test_bench_normcdf_batch_1m_gpu(benchmark, rng=np.random.default_rng(0)):
    x = cp.asarray(rng.uniform(-3, 3, 1_000_000))
    benchmark(normcdf, x)

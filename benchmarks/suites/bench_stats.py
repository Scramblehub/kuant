"""Benchmarks for kuant.stats — rolling primitives, risk metrics, tail cluster."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import (
    dfa,
    hurstrs,
    rollcalmar,
    rollcorr,
    rollema,
    rollmax,
    rollmdd,
    rollmean,
    rollsharpe,
    rollsortino,
    rollstd,
    tailindex,
    zscore,
)

try:
    import cupy as cp

    _HAS_GPU = cp.cuda.is_available()
except (ImportError, RuntimeError):
    cp = None
    _HAS_GPU = False


# ---------- Rolling primitives ---------------------------------------------


def test_bench_rollmean_10k_w252(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(10_000)
    benchmark(rollmean, x, 252)


def test_bench_rollstd_10k_w252(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(10_000)
    benchmark(rollstd, x, 252)


def test_bench_rollema_10k(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(10_000)
    benchmark(rollema, x, 21)


def test_bench_rollmax_10k_w63(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(10_000)
    benchmark(rollmax, x, 63)


def test_bench_rollcorr_10k_w252(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(10_000)
    y = 0.5 * x + rng.standard_normal(10_000)
    benchmark(rollcorr, x, y, 252)


def test_bench_zscore_10k_w252(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(10_000)
    benchmark(zscore, x, 252)


# ---------- Risk metrics ---------------------------------------------------


def test_bench_rollsharpe_5k_w252(benchmark, rng=np.random.default_rng(0)):
    r = rng.normal(0.001, 0.01, 5_000)
    benchmark(rollsharpe, r, 252, 252)


def test_bench_rollsortino_5k_w252(benchmark, rng=np.random.default_rng(0)):
    r = rng.normal(0.001, 0.01, 5_000)
    benchmark(rollsortino, r, 252, 252)


def test_bench_rollmdd_2k_w63(benchmark, rng=np.random.default_rng(0)):
    """rollmdd is O(n*w) — the expensive risk metric to profile."""
    r = rng.normal(0.001, 0.01, 2_000)
    benchmark(rollmdd, r, 63)


def test_bench_rollcalmar_2k_w63(benchmark, rng=np.random.default_rng(0)):
    r = rng.normal(0.001, 0.01, 2_000)
    benchmark(rollcalmar, r, 63, 252)


# ---------- Tail cluster ---------------------------------------------------


def test_bench_hurstrs_5k(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(5_000)
    benchmark(hurstrs, x)


def test_bench_dfa_5k(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(5_000)
    benchmark(dfa, x)


def test_bench_tailindex_10k(benchmark, rng=np.random.default_rng(0)):
    x = (1 - rng.uniform(size=10_000)) ** (-0.5)
    benchmark(tailindex, x)


# ---------- GPU ------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA not available")
def test_bench_rollmean_100k_w252_gpu(benchmark, rng=np.random.default_rng(0)):
    x = cp.asarray(rng.standard_normal(100_000))
    benchmark(rollmean, x, 252)


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA not available")
def test_bench_rollcorr_100k_w252_gpu(benchmark, rng=np.random.default_rng(0)):
    x = cp.asarray(rng.standard_normal(100_000))
    y = cp.asarray(0.5 * cp.asnumpy(x) + rng.standard_normal(100_000))
    benchmark(rollcorr, x, y, 252)

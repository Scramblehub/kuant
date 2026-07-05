"""Benchmarks for kuant.signals — winsorize, neutralize, icdecay."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from kuant.signals import icdecay, neutralize, winsorize  # noqa: E402


# ---------- winsorize ---------------------------------------------------


def test_bench_winsorize_1d_100k(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(100_000)
    benchmark(winsorize, x, 0.01, 0.99)


def test_bench_winsorize_2d_per_row(benchmark, rng=np.random.default_rng(0)):
    """2D per-row: 500 dates × 1000 names — classic factor-score winsorize."""
    x = rng.standard_normal((500, 1000))
    benchmark(winsorize, x, 0.01, 0.99, True)


def test_bench_winsorize_2d_per_column(benchmark, rng=np.random.default_rng(0)):
    """2D per-column: 5000 dates × 100 names — time-series noise clip."""
    x = rng.standard_normal((5000, 100))
    benchmark(winsorize, x, 0.01, 0.99, False)


# ---------- neutralize --------------------------------------------------


def test_bench_neutralize_3_factors_2k(benchmark, rng=np.random.default_rng(0)):
    T = 2000
    signal = rng.standard_normal(T)
    factors = {f"f{i}": rng.standard_normal(T) for i in range(3)}
    benchmark(neutralize, signal, factors)


def test_bench_neutralize_10_factors_5k(benchmark, rng=np.random.default_rng(0)):
    T = 5000
    signal = rng.standard_normal(T)
    X = rng.standard_normal((T, 10))
    benchmark(neutralize, signal, X)


# ---------- icdecay -----------------------------------------------------


def test_bench_icdecay_4_horizons_2k(benchmark, rng=np.random.default_rng(0)):
    T = 2000
    signal = rng.standard_normal(T)
    forward_ret = 0.05 * signal + rng.standard_normal(T) * 0.1
    benchmark(icdecay, signal, forward_ret, (1, 5, 21, 63))


def test_bench_icdecay_10_horizons_10k(benchmark, rng=np.random.default_rng(0)):
    T = 10_000
    signal = rng.standard_normal(T)
    forward_ret = rng.standard_normal(T)
    horizons = (1, 2, 5, 10, 21, 42, 63, 126, 189, 252)
    benchmark(icdecay, signal, forward_ret, horizons)

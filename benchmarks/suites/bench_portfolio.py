"""Benchmarks for kuant.portfolio."""

from __future__ import annotations

import numpy as np

from kuant.portfolio import contribution, drawdown, sharperatio, sortinoratio


# ---------- drawdown -----------------------------------------------------


def test_bench_drawdown_10k(benchmark, rng=np.random.default_rng(0)):
    equity = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 10_000))
    benchmark(drawdown, equity)


def test_bench_drawdown_100k(benchmark, rng=np.random.default_rng(0)):
    equity = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 100_000))
    benchmark(drawdown, equity)


# ---------- sharperatio + sortinoratio ---------------------------------


def test_bench_sharperatio_10k(benchmark, rng=np.random.default_rng(0)):
    r = rng.normal(0.001, 0.01, 10_000)
    benchmark(sharperatio, r, 252, 0.0)


def test_bench_sortinoratio_10k(benchmark, rng=np.random.default_rng(0)):
    r = rng.normal(0.001, 0.01, 10_000)
    benchmark(sortinoratio, r, 252, 0.0)


# ---------- contribution ------------------------------------------------


def test_bench_contribution_2y_500names(benchmark, rng=np.random.default_rng(0)):
    """~2 years of daily P&L for 500 names."""
    T, N = 504, 500
    positions = rng.normal(0, 1, (T, N))
    returns = rng.normal(0, 0.02, (T, N))
    benchmark(contribution, positions, returns)


def test_bench_contribution_with_groups(benchmark, rng=np.random.default_rng(0)):
    T, N = 252, 200
    positions = rng.normal(0, 1, (T, N))
    returns = rng.normal(0, 0.02, (T, N))
    groups = rng.choice(["tech", "energy", "financials", "utility"], size=N)
    benchmark(contribution, positions, returns, groups)

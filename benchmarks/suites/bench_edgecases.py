"""Benchmarks for kuant.edgecases — nanpolicies, delistedhandling, outlierpolicy."""

from __future__ import annotations

import numpy as np

from kuant.edgecases import (
    full_recovery_check,
    hold_last_price,
    nanpolicies,
    outlierpolicy,
    zero_after_delist,
)


# ---------- nanpolicies ---------------------------------------------------


def test_bench_forwardfill_100k_1d(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(100_000)
    x[rng.uniform(size=100_000) < 0.2] = np.nan
    benchmark(nanpolicies.forwardfill, x)


def test_bench_interpolate_100k_1d(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(100_000)
    x[rng.uniform(size=100_000) < 0.2] = np.nan
    benchmark(nanpolicies.interpolate, x)


def test_bench_forwardfill_2d_10k_x_50(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal((10_000, 50))
    x[rng.uniform(size=x.shape) < 0.1] = np.nan
    benchmark(nanpolicies.forwardfill, x)


def test_bench_dropcolumn_10k_x_100(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal((10_000, 100))
    # Roughly half the columns have low coverage.
    for c in range(50):
        x[:, c][rng.uniform(size=10_000) < 0.8] = np.nan
    benchmark(nanpolicies.dropcolumn, x, 0.5)


def test_bench_skipna_2d(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal((10_000, 20))
    x[rng.uniform(size=x.shape) < 0.05] = np.nan
    benchmark(nanpolicies.skipna, x)


# ---------- delistedhandling ---------------------------------------------


def test_bench_zero_after_delist_10k(benchmark, rng=np.random.default_rng(0)):
    prices = 100.0 + rng.standard_normal(10_000)
    benchmark(zero_after_delist, prices, 5000)


def test_bench_hold_last_price_10k(benchmark, rng=np.random.default_rng(0)):
    prices = 100.0 + rng.standard_normal(10_000)
    # max_hold_days set very high to avoid the warning noise in bench.
    benchmark(hold_last_price, prices, 5000, 10_000)


def test_bench_full_recovery_check_1k_universe(benchmark, rng=np.random.default_rng(0)):
    universe = np.array([f"T{i:04d}" for i in range(1000)])
    known = universe[rng.integers(0, 1000, size=50)]
    benchmark(full_recovery_check, universe, known)


# ---------- outlierpolicy -------------------------------------------------


def test_bench_outlierpolicy_mad_50k(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(50_000)
    benchmark(outlierpolicy, x, "mad", 3.0)


def test_bench_outlierpolicy_iqr_50k(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(50_000)
    benchmark(outlierpolicy, x, "iqr", 1.5)


def test_bench_outlierpolicy_zscore_50k(benchmark, rng=np.random.default_rng(0)):
    x = rng.standard_normal(50_000)
    benchmark(outlierpolicy, x, "zscore", 3.0)

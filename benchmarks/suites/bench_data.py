"""Benchmarks for kuant.data — align, baragg, corpaction."""

from __future__ import annotations

import numpy as np

from kuant.data import align, baragg, corpaction


# ---------- align ---------------------------------------------------------


def test_bench_align_inner_2way_1k(benchmark, rng=np.random.default_rng(0)):
    """Inner join two 1000-entry series with 90% overlap."""
    idx_a = np.arange(1000)
    val_a = rng.standard_normal(1000)
    idx_b = np.arange(100, 1100)
    val_b = rng.standard_normal(1000)
    benchmark(align, (idx_a, val_a), (idx_b, val_b), method="inner")


def test_bench_align_outer_3way_5k(benchmark, rng=np.random.default_rng(0)):
    """Outer join three 5k series with partial overlap."""
    a = (np.arange(5000), rng.standard_normal(5000))
    b = (np.arange(1000, 6000), rng.standard_normal(5000))
    c = (np.arange(2000, 7000), rng.standard_normal(5000))
    benchmark(align, a, b, c, method="outer")


def test_bench_align_forward_2way_10k(benchmark, rng=np.random.default_rng(0)):
    """Forward-fill join — fills 40% of positions."""
    dense = (np.arange(10_000), rng.standard_normal(10_000))
    sparse_idx = np.arange(0, 10_000, 3)  # every third position
    sparse = (sparse_idx, rng.standard_normal(len(sparse_idx)))
    benchmark(align, dense, sparse, method="forward")


# ---------- baragg -------------------------------------------------------


def test_bench_baragg_10k_ticks_100_buckets(benchmark, rng=np.random.default_rng(0)):
    """10k ticks aggregated into 100 buckets (100 ticks per bar)."""
    bucket = np.arange(10_000) // 100
    close = 100.0 + np.cumsum(rng.standard_normal(10_000)) * 0.01
    volume = rng.integers(1, 1000, size=10_000)
    benchmark(baragg, bucket, close, volume=volume)


def test_bench_baragg_full_ohlc_50k(benchmark, rng=np.random.default_rng(0)):
    """50k minute-bars aggregated into 5-minute bars (10k output bars)."""
    n = 50_000
    bucket = np.arange(n) // 5
    close = 100.0 + rng.standard_normal(n) * 0.5
    open_ = close + rng.standard_normal(n) * 0.1
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n) * 0.05)
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n) * 0.05)
    volume = rng.integers(1, 1_000_000, size=n)
    benchmark(baragg, bucket, close, open=open_, high=high, low=low, volume=volume)


# ---------- corpaction ---------------------------------------------------


def test_bench_corpaction_split_only_10y(benchmark, rng=np.random.default_rng(0)):
    """~10 years daily prices, 2 splits."""
    n = 2520
    prices = 100.0 + np.cumsum(rng.standard_normal(n)) * 0.5
    benchmark(
        corpaction,
        prices,
        split_positions=[500, 1200],
        split_ratios=[2.0, 3.0],
        mode="split_only",
        direction="backward",
    )


def test_bench_corpaction_total_return_10y_20div(benchmark, rng=np.random.default_rng(0)):
    """10y daily with 2 splits + 20 quarterly dividends (total-return backward)."""
    n = 2520
    prices = 100.0 + np.cumsum(rng.standard_normal(n)) * 0.5
    prices = np.abs(prices) + 1.0  # keep positive
    div_positions = np.arange(63, 2520, 63)[:20]
    div_amounts = np.full(20, 0.5)
    benchmark(
        corpaction,
        prices,
        split_positions=[500, 1200],
        split_ratios=[2.0, 3.0],
        dividend_positions=div_positions,
        dividend_amounts=div_amounts,
        mode="total_return",
        direction="backward",
    )

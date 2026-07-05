"""Benchmarks for kuant.text - occparse, secformparse, cusipvalidate."""

from __future__ import annotations

import random

from kuant.text import cusipvalidate, occparse, secformparse


def test_bench_occparse_single(benchmark):
    benchmark(occparse, "AAPL240119C00150000")


def test_bench_occparse_batch_1000(benchmark):
    """Parse 1000 varied option symbols."""
    rng = random.Random(0)
    roots = ["AAPL", "SPY", "TSLA", "QQQ", "META", "MSFT", "NVDA", "AMD"]
    symbols = []
    for _ in range(1000):
        root = rng.choice(roots)
        yy = rng.randint(24, 27)
        mm = rng.randint(1, 12)
        dd = rng.randint(1, 28)
        right = rng.choice(["C", "P"])
        strike = rng.randint(1_000, 900_000)
        symbols.append(f"{root}{yy:02d}{mm:02d}{dd:02d}{right}{strike:08d}")

    def _run():
        for s in symbols:
            occparse(s)

    benchmark(_run)


def test_bench_secformparse_single(benchmark):
    benchmark(secformparse, "10-K/A")


def test_bench_secformparse_batch_1000(benchmark):
    rng = random.Random(0)
    forms = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A", "13F-HR", "3", "4", "5", "20-F"]
    pool = [f + ("/A" if rng.random() < 0.15 else "") for f in forms]
    inputs = [rng.choice(pool) for _ in range(1000)]

    def _run():
        for f in inputs:
            secformparse(f)

    benchmark(_run)


def test_bench_cusipvalidate_single(benchmark):
    benchmark(cusipvalidate, "037833100")


def test_bench_cusipvalidate_batch_1000(benchmark):
    """Validate 1000 varied CUSIPs (mix of valid + invalid)."""
    rng = random.Random(0)
    known_valid = ["037833100", "594918104", "023135106", "88160R101", "02079K305", "30303M102"]
    inputs = []
    for _ in range(1000):
        if rng.random() < 0.7:
            inputs.append(rng.choice(known_valid))
        else:
            # Corrupt check digit.
            base = rng.choice(known_valid)
            inputs.append(base[:8] + str((int(base[-1]) + 1) % 10))

    def _run():
        for c in inputs:
            cusipvalidate(c)

    benchmark(_run)

"""Benchmarks for kuant.topology — persistent-homology + dispersion signals.

persistenthomology cost scales with n_points^3 in the worst case
(sparse Rips), so windows in the 50-200 range dominate real usage.
bettiseries wraps it in a rolling loop — the benchmark measures the
per-anchor cost. dispersioncollapse is pure numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ripser")
pytest.importorskip("persim")

from kuant.topology import (  # noqa: E402
    bettiseries,
    dispersioncollapse,
    persistenthomology,
    wasserstein,
)


# ---------- persistenthomology --------------------------------------------


def test_bench_persistenthomology_pointcloud_100(benchmark, rng=np.random.default_rng(0)):
    """Small 2D point cloud — realistic per-window size for bettiseries."""
    cloud = rng.normal(size=(100, 2))
    benchmark(persistenthomology, cloud, 1)


def test_bench_persistenthomology_pointcloud_300(benchmark, rng=np.random.default_rng(0)):
    """Medium point cloud — where per-anchor cost starts to bite."""
    cloud = rng.normal(size=(300, 2))
    benchmark(persistenthomology, cloud, 1)


def test_bench_persistenthomology_takens_1d(benchmark, rng=np.random.default_rng(0)):
    """1D series → Takens embed → PH. Common ingestion path."""
    x = np.sin(np.linspace(0, 20 * np.pi, 200))
    benchmark(persistenthomology, x, 1, 3, 6)


# ---------- bettiseries ---------------------------------------------------


def test_bench_bettiseries_len500_w100(benchmark, rng=np.random.default_rng(0)):
    """Realistic rolling PH scan over a 500-bar series."""
    x = rng.standard_normal(500)
    benchmark(bettiseries, x, 100, 1, 3, 6, 0.1, 25)  # stride=25 keeps this under a second


# ---------- wasserstein --------------------------------------------------


def test_bench_wasserstein_small(benchmark, rng=np.random.default_rng(0)):
    a = np.column_stack([rng.uniform(0, 1, 20), rng.uniform(1, 2, 20)])
    b = np.column_stack([rng.uniform(0, 1, 20), rng.uniform(1, 2, 20)])
    benchmark(wasserstein, a, b, "wasserstein")


def test_bench_bottleneck_small(benchmark, rng=np.random.default_rng(0)):
    a = np.column_stack([rng.uniform(0, 1, 20), rng.uniform(1, 2, 20)])
    b = np.column_stack([rng.uniform(0, 1, 20), rng.uniform(1, 2, 20)])
    benchmark(wasserstein, a, b, "bottleneck")


def test_bench_sliced_wasserstein_100pts(benchmark, rng=np.random.default_rng(0)):
    a = np.column_stack([rng.uniform(0, 1, 100), rng.uniform(1, 2, 100)])
    b = np.column_stack([rng.uniform(0, 1, 100), rng.uniform(1, 2, 100)])
    benchmark(wasserstein, a, b, "sliced_wasserstein", 50)


# ---------- dispersioncollapse -------------------------------------------


def test_bench_dispersioncollapse_2000x30(benchmark, rng=np.random.default_rng(0)):
    """Typical panel size: 2000 bars, 30 names."""
    R = rng.normal(size=(2000, 30))
    benchmark(dispersioncollapse, R, 63, 0.20, 5)

"""Test suite for kuant.core.tcdf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as sp_t

from kuant.core import tcdf


@pytest.mark.parametrize(
    "x, df",
    [
        (0.0, 3.0),
        (1.0, 5.0),
        (-1.5, 10.0),
        (3.0, 30.0),
        (0.5, 100.0),
        (2.0, 2.0),
        (-5.0, 1.0),
        (0.0, 1e6),
        (5.0, 3.0),  # deep upper tail
        (-10.0, 5.0),  # deep lower tail
    ],
)
def test_matches_scipy(x, df):
    ours = tcdf(x, df)
    ref = sp_t.cdf(x, df)
    assert abs(ours - ref) < 1e-13


def test_batched(rng):
    xs = rng.uniform(-5, 5, 200)
    dfs = rng.uniform(2, 100, 200)
    ours = tcdf(xs, dfs)
    ref = sp_t.cdf(xs, dfs)
    np.testing.assert_allclose(ours, ref, atol=1e-13)


def test_at_zero_is_half():
    for df in [1.0, 3.0, 10.0, 100.0]:
        assert abs(tcdf(0.0, df) - 0.5) < 1e-14


def test_symmetry():
    """tcdf(-x) = 1 - tcdf(x)."""
    for x in [0.5, 1.5, 3.0]:
        for df in [3.0, 10.0, 30.0]:
            assert abs(tcdf(-x, df) - (1 - tcdf(x, df))) < 1e-14


def test_monotonic(rng):
    xs = np.sort(rng.uniform(-3, 3, 20))
    df = 5.0
    vals = tcdf(xs, df)
    assert np.all(np.diff(vals) >= 0)


def test_range_0_1(rng):
    xs = rng.uniform(-10, 10, 100)
    dfs = rng.uniform(1, 100, 100)
    vals = tcdf(xs, dfs)
    assert np.all(vals >= 0) and np.all(vals <= 1)


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    xs = rng.uniform(-3, 3, 50)
    dfs = rng.uniform(2, 30, 50)
    r_cpu = tcdf(xs, dfs)
    r_gpu = cp.asnumpy(tcdf(cp.asarray(xs), cp.asarray(dfs)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

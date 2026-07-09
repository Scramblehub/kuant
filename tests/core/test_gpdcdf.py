"""Test suite for kuant.core.gpdcdf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from kuant.core import gpdcdf


@pytest.mark.parametrize(
    "x, xi, scale",
    [
        (0.5, 0.2, 1.0),
        (0.5, 0.0, 1.0),
        (0.5, -0.3, 1.0),
        (2.0, 0.5, 2.0),
        (5.0, 1.0, 1.0),
        (0.1, 0.7, 0.5),
    ],
)
def test_matches_scipy(x, xi, scale):
    ours = gpdcdf(x, xi, scale)
    ref = genpareto.cdf(x, xi, loc=0, scale=scale)
    assert abs(ours - ref) < 1e-13


def test_zero_below_support():
    for xi in [-0.5, 0.0, 0.5]:
        assert gpdcdf(-1.0, xi, 1.0) == 0.0


def test_one_at_upper_bound_when_xi_negative():
    xi, scale = -0.5, 1.0
    upper = -scale / xi
    # Any x above upper bound should give 1
    assert gpdcdf(upper + 0.5, xi, scale) == 1.0


def test_monotonic_in_x(rng):
    xs = np.sort(rng.uniform(0, 3, 20))
    vals = gpdcdf(xs, 0.3, 1.0)
    assert np.all(np.diff(vals) >= 0)


def test_range_0_1(rng):
    xs = rng.uniform(0, 10, 100)
    xis = rng.uniform(-0.4, 1.0, 100)
    scales = rng.uniform(0.5, 3.0, 100)
    vals = gpdcdf(xs, xis, scales)
    assert np.all(vals >= 0) and np.all(vals <= 1)


def test_exponential_limit():
    x, scale = 1.5, 2.0
    expected = 1.0 - np.exp(-x / scale)
    assert abs(gpdcdf(x, 0.0, scale) - expected) < 1e-14


def test_batched(rng):
    xs = rng.uniform(0, 5, 100)
    xis = rng.uniform(-0.4, 1.0, 100)
    scales = rng.uniform(0.5, 3.0, 100)
    ours = gpdcdf(xs, xis, scales)
    ref = genpareto.cdf(xs, xis, loc=0, scale=scales)
    np.testing.assert_allclose(ours, ref, atol=1e-13)


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    xs = rng.uniform(0, 3, 50)
    xis = rng.uniform(-0.3, 0.5, 50)
    scales = rng.uniform(0.5, 2.0, 50)
    r_cpu = gpdcdf(xs, xis, scales)
    r_gpu = cp.asnumpy(gpdcdf(cp.asarray(xs), cp.asarray(xis), cp.asarray(scales)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

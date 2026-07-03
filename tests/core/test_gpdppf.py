'''Test suite for kuant.core.gpdppf.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from kuant.core import gpdcdf, gpdppf


@pytest.mark.parametrize(
    'p, xi, scale',
    [
        (0.1, 0.5, 1.0), (0.5, 0.0, 1.0), (0.9, -0.3, 1.0),
        (0.99, 0.5, 2.0), (0.05, 0.2, 1.0),
    ],
)
def test_matches_scipy(p, xi, scale):
    ours = gpdppf(p, xi, scale)
    ref = genpareto.ppf(p, xi, loc=0, scale=scale)
    assert abs(ours - ref) < 1e-13


def test_round_trip(rng):
    ps = rng.uniform(0.01, 0.99, 100)
    xis = rng.uniform(-0.3, 0.5, 100)
    scales = rng.uniform(0.5, 2.0, 100)
    xs = gpdppf(ps, xis, scales)
    ps_back = gpdcdf(xs, xis, scales)
    np.testing.assert_allclose(ps_back, ps, atol=1e-13)


def test_p_zero_is_zero():
    for xi in [-0.3, 0.0, 0.5]:
        assert gpdppf(0.0, xi, 1.0) == 0.0


def test_p_one_returns_upper_support():
    # xi >= 0: unbounded → +inf
    assert gpdppf(1.0, 0.5, 1.0) == np.inf
    assert gpdppf(1.0, 0.0, 1.0) == np.inf
    # xi < 0: -scale/xi
    xi, scale = -0.5, 1.0
    upper = -scale / xi
    assert abs(gpdppf(1.0, xi, scale) - upper) < 1e-14


def test_out_of_range_returns_nan():
    assert np.isnan(gpdppf(-0.1, 0.3, 1.0))
    assert np.isnan(gpdppf(1.5, 0.3, 1.0))
    assert np.isnan(gpdppf(0.5, 0.3, -1.0))   # scale ≤ 0
    assert np.isnan(gpdppf(np.nan, 0.3, 1.0))


def test_exponential_limit():
    p, scale = 0.5, 2.0
    expected = -scale * np.log(1 - p)
    assert abs(gpdppf(p, 0.0, scale) - expected) < 1e-14


def test_batched(rng):
    ps = rng.uniform(0.01, 0.99, 100)
    xis = rng.uniform(-0.3, 0.5, 100)
    scales = rng.uniform(0.5, 2.0, 100)
    ours = gpdppf(ps, xis, scales)
    ref = genpareto.ppf(ps, xis, loc=0, scale=scales)
    np.testing.assert_allclose(ours, ref, atol=1e-13)


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    ps = rng.uniform(0.05, 0.95, 30)
    xis = rng.uniform(-0.3, 0.5, 30)
    scales = rng.uniform(0.5, 2.0, 30)
    r_cpu = gpdppf(ps, xis, scales)
    r_gpu = cp.asnumpy(gpdppf(cp.asarray(ps), cp.asarray(xis), cp.asarray(scales)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

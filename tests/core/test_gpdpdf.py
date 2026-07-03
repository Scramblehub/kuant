'''Test suite for kuant.core.gpdpdf.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from kuant.core import gpdpdf


@pytest.mark.parametrize(
    'x, xi, scale',
    [
        (0.5, 0.2, 1.0),   # positive shape
        (0.5, 0.0, 1.0),   # exponential limit
        (0.5, -0.3, 1.0),  # negative shape (bounded)
        (2.0, 0.5, 2.0),
        (5.0, 1.0, 1.0),   # deep tail
        (0.1, 0.7, 0.5),
    ],
)
def test_matches_scipy(x, xi, scale):
    ours = gpdpdf(x, xi, scale)
    ref = genpareto.pdf(x, xi, loc=0, scale=scale)
    assert abs(ours - ref) < 1e-13


def test_zero_below_support():
    '''PDF is zero for x < 0 across any xi/scale.'''
    for xi in [-0.5, 0.0, 0.5]:
        assert gpdpdf(-1.0, xi, 1.0) == 0.0


def test_zero_above_upper_bound_when_xi_negative():
    '''For xi < 0, upper bound is -scale/xi. Above it, PDF = 0.'''
    xi, scale = -0.5, 1.0
    upper = -scale / xi
    assert gpdpdf(upper + 0.5, xi, scale) == 0.0


def test_exponential_limit_at_xi_zero():
    '''gpdpdf(x, 0, scale) = (1/scale) * exp(-x/scale).'''
    x, scale = 1.5, 2.0
    expected = (1.0 / scale) * np.exp(-x / scale)
    assert abs(gpdpdf(x, 0.0, scale) - expected) < 1e-14


def test_batched(rng):
    xs = rng.uniform(0, 5, 100)
    xis = rng.uniform(-0.4, 1.0, 100)
    scales = rng.uniform(0.5, 3.0, 100)
    ours = gpdpdf(xs, xis, scales)
    ref = genpareto.pdf(xs, xis, loc=0, scale=scales)
    np.testing.assert_allclose(ours, ref, atol=1e-13)


def test_broadcast_scalar_scale():
    xs = np.array([0.5, 1.0, 2.0])
    result = gpdpdf(xs, 0.3, 1.0)
    for i, x in enumerate(xs):
        assert abs(result[i] - gpdpdf(float(x), 0.3, 1.0)) < 1e-14


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    xs = rng.uniform(0, 3, 50)
    xis = rng.uniform(-0.3, 0.5, 50)
    scales = rng.uniform(0.5, 2.0, 50)
    r_cpu = gpdpdf(xs, xis, scales)
    r_gpu = cp.asnumpy(gpdpdf(cp.asarray(xs), cp.asarray(xis), cp.asarray(scales)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

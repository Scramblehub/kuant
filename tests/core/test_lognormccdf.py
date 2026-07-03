'''Test suite for kuant.core.lognormccdf.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import lognormccdf, lognormcdf


@pytest.mark.parametrize(
    'x', [-40.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 40.0]
)
def test_matches_scipy_logsf(x):
    ours = lognormccdf(x)
    ref = norm.logsf(x)
    if abs(ref) > 100:
        assert abs(ours - ref) / abs(ref) < 1e-8
    else:
        assert abs(ours - ref) < 1e-12


def test_symmetry_with_lognormcdf():
    '''lognormccdf(x) == lognormcdf(-x) by construction.'''
    for x in [-3.0, 0.0, 3.0, 10.0]:
        assert abs(lognormccdf(x) - lognormcdf(-x)) < 1e-14


def test_x_zero_is_neg_log_2():
    assert abs(lognormccdf(0.0) - (-np.log(2))) < 1e-14


def test_extreme_positive_still_finite():
    '''log(1-Φ(x)) for large x — naive log(1 - normcdf(x)) = log(0) = -inf.'''
    for x in [40.0, 50.0, 100.0]:
        result = lognormccdf(x)
        assert np.isfinite(result)


def test_batched(rng):
    x = rng.uniform(-20, 20, 100)
    result = lognormccdf(x)
    for i, xi in enumerate(x):
        assert abs(result[i] - lognormccdf(float(xi))) < 1e-12


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x = rng.uniform(-20, 20, 100)
    r_cpu = lognormccdf(x)
    r_gpu = cp.asnumpy(lognormccdf(cp.asarray(x)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

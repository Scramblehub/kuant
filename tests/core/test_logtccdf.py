'''Test suite for kuant.core.logtccdf.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as sp_t

from kuant.core import logtccdf, logtcdf


@pytest.mark.parametrize(
    'x, df',
    [
        (-5.0, 3.0), (-1.0, 10.0), (0.0, 5.0),
        (1.0, 10.0), (3.0, 5.0), (5.0, 3.0),
        (100.0, 3.0), (300.0, 10.0), (1000.0, 3.0),
    ],
)
def test_matches_scipy_logsf(x, df):
    ours = logtccdf(x, df)
    ref = sp_t.logsf(x, df)
    assert abs(ours - ref) < 1e-12


def test_symmetry_with_logtcdf():
    '''logtccdf(x) = logtcdf(-x) by construction.'''
    for x in [-3.0, 0.0, 3.0, 10.0]:
        for df in [3.0, 5.0, 10.0]:
            assert abs(logtccdf(x, df) - logtcdf(-x, df)) < 1e-14


def test_at_zero_is_neg_log_2():
    assert abs(logtccdf(0.0, 5.0) - (-np.log(2))) < 1e-14


def test_extreme_positive_stays_finite():
    for x in [1000.0, 10000.0]:
        for df in [3.0, 5.0]:
            result = logtccdf(x, df)
            assert np.isfinite(result)


def test_batched(rng):
    xs = rng.uniform(-10, 10, 100)
    dfs = rng.uniform(2, 30, 100)
    ours = logtccdf(xs, dfs)
    ref = sp_t.logsf(xs, dfs)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    xs = rng.uniform(-5, 5, 30)
    dfs = rng.uniform(2, 30, 30)
    r_cpu = logtccdf(xs, dfs)
    r_gpu = cp.asnumpy(logtccdf(cp.asarray(xs), cp.asarray(dfs)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

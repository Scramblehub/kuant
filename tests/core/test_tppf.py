'''Test suite for kuant.core.tppf.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as sp_t

from kuant.core import tcdf, tppf


@pytest.mark.parametrize(
    'p, df',
    [
        (0.01, 3.0), (0.05, 5.0), (0.5, 10.0),
        (0.95, 30.0), (0.99, 100.0),
        (0.001, 5.0),  # deep tail
        (0.999, 3.0),  # deep upper tail
    ],
)
def test_matches_scipy(p, df):
    ours = tppf(p, df)
    ref = sp_t.ppf(p, df)
    assert abs(ours - ref) < 1e-10


def test_round_trip_via_tcdf(rng):
    ps = rng.uniform(0.01, 0.99, 100)
    dfs = rng.uniform(2, 50, 100)
    xs = tppf(ps, dfs)
    ps_back = tcdf(xs, dfs)
    np.testing.assert_allclose(ps_back, ps, atol=1e-10)


def test_p_half_is_zero():
    for df in [1.0, 3.0, 10.0, 100.0]:
        assert abs(tppf(0.5, df)) < 1e-14


def test_symmetry():
    '''tppf(1 - p, df) = -tppf(p, df).'''
    for p in [0.05, 0.1, 0.25]:
        for df in [3.0, 10.0, 30.0]:
            assert abs(tppf(1 - p, df) + tppf(p, df)) < 1e-9


def test_boundary_p_zero_is_neg_inf():
    assert tppf(0.0, 5.0) == -np.inf


def test_boundary_p_one_is_pos_inf():
    assert tppf(1.0, 5.0) == np.inf


def test_out_of_range_returns_nan():
    assert np.isnan(tppf(-0.1, 5.0))
    assert np.isnan(tppf(1.5, 5.0))
    assert np.isnan(tppf(0.5, -1.0))   # invalid df
    assert np.isnan(tppf(np.nan, 5.0))


def test_batched(rng):
    ps = rng.uniform(0.01, 0.99, 100)
    dfs = rng.uniform(2, 100, 100)
    ours = tppf(ps, dfs)
    ref = sp_t.ppf(ps, dfs)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    ps = rng.uniform(0.05, 0.95, 30)
    dfs = rng.uniform(2, 30, 30)
    r_cpu = tppf(ps, dfs)
    r_gpu = cp.asnumpy(tppf(cp.asarray(ps), cp.asarray(dfs)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

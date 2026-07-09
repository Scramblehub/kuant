"""Test suite for kuant.core.tpdf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as sp_t

from kuant.core import tpdf


@pytest.mark.parametrize(
    "x, df",
    [
        (0.0, 3.0),
        (1.0, 5.0),
        (-1.5, 10.0),
        (3.0, 30.0),
        (0.5, 100.0),
        (2.0, 2.0),
        (-5.0, 1.0),  # Cauchy limit (df=1)
        (0.0, 1e6),  # near-Gaussian limit
    ],
)
def test_matches_scipy(x, df):
    ours = tpdf(x, df)
    ref = sp_t.pdf(x, df)
    # Extreme df (>1e5) uses gammaln-based Gaussian-limit computation vs
    # scipy's alternate; both are valid to ~1e-10. Ordinary df: 1e-13.
    tol = 1e-9 if df > 1e5 else 1e-13
    assert abs(ours - ref) < tol


def test_batched_matches_scipy(rng):
    xs = rng.uniform(-5, 5, 200)
    dfs = rng.uniform(2, 100, 200)
    ours = tpdf(xs, dfs)
    ref = sp_t.pdf(xs, dfs)
    np.testing.assert_allclose(ours, ref, atol=1e-13)


def test_broadcast_scalar_df():
    xs = np.array([-2.0, 0.0, 2.0])
    result = tpdf(xs, 5.0)
    for i, x in enumerate(xs):
        assert abs(result[i] - tpdf(float(x), 5.0)) < 1e-14


def test_broadcast_2d():
    xs = np.array([[-1.0, 0.0, 1.0]])  # (1, 3)
    dfs = np.array([[3.0], [10.0]])  # (2, 1)
    result = tpdf(xs, dfs)
    assert result.shape == (2, 3)


def test_symmetric_about_zero():
    for x, df in [(1.5, 5.0), (2.0, 10.0), (0.7, 30.0)]:
        assert abs(tpdf(x, df) - tpdf(-x, df)) < 1e-15


def test_int_promoted_to_float64():
    result = tpdf(0, 5)
    assert isinstance(result, float)
    assert abs(result - tpdf(0.0, 5.0)) < 1e-14


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    xs = rng.uniform(-3, 3, 50)
    dfs = rng.uniform(2, 30, 50)
    r_cpu = tpdf(xs, dfs)
    r_gpu = cp.asnumpy(tpdf(cp.asarray(xs), cp.asarray(dfs)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

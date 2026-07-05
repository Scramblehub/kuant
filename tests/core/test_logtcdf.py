"""Test suite for kuant.core.logtcdf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as sp_t

from kuant.core import logtcdf


@pytest.mark.parametrize(
    "x, df",
    [
        (-5.0, 3.0),
        (-3.0, 5.0),
        (-1.0, 10.0),
        (0.0, 3.0),
        (1.0, 10.0),
        (3.0, 5.0),
        (5.0, 3.0),
        (-100.0, 3.0),
        (-300.0, 10.0),
        (-1000.0, 3.0),  # deep tail
    ],
)
def test_matches_scipy(x, df):
    ours = logtcdf(x, df)
    ref = sp_t.logcdf(x, df)
    assert abs(ours - ref) < 1e-12


def test_at_zero_is_neg_log_2():
    for df in [1.0, 3.0, 10.0, 100.0]:
        assert abs(logtcdf(0.0, df) - (-np.log(2))) < 1e-14


def test_batched(rng):
    xs = rng.uniform(-10, 10, 100)
    dfs = rng.uniform(2, 30, 100)
    ours = logtcdf(xs, dfs)
    ref = sp_t.logcdf(xs, dfs)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_extreme_negative_stays_finite():
    """Naive log(tcdf) underflows at extreme |x|; ours stays finite."""
    for x in [-1000.0, -10000.0]:
        for df in [3.0, 5.0]:
            result = logtcdf(x, df)
            assert np.isfinite(result), f"x={x} df={df} gave {result}"


def test_nan_propagation():
    result = logtcdf(np.array([np.nan]), 5.0)
    assert np.isnan(result[0])


def test_nonpositive_df_raises():
    import pytest
    from kuant.errors import KuantValueError

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        logtcdf(1.0, -1.0)
    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        logtcdf(1.0, 0.0)


def test_nan_df_returns_nan():
    """NaN df is not caught by the (<= 0).any() guard; downstream
    betainc returns NaN which propagates through the log."""
    assert np.isnan(logtcdf(1.0, np.nan))


def test_no_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for x in [-1000.0, -1.0, 0.0, 1.0, 100.0]:
            _ = logtcdf(x, 5.0)


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    xs = rng.uniform(-5, 5, 30)
    dfs = rng.uniform(2, 30, 30)
    r_cpu = logtcdf(xs, dfs)
    r_gpu = cp.asnumpy(logtcdf(cp.asarray(xs), cp.asarray(dfs)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)

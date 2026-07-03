'''Test suite for kuant.core.lognormcdf.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import lognormcdf


# ---------------------------------------------------------------------------
# 1. Match scipy.stats.norm.logcdf across the whole range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'x', [-40.0, -20.0, -10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0, 20.0, 40.0]
)
def test_matches_scipy(x):
    ours = lognormcdf(x)
    ref = norm.logcdf(x)
    # For extreme tails, allow tiny truncation error from Mills asymptotic
    if abs(ref) > 100:
        assert abs(ours - ref) / abs(ref) < 1e-8
    else:
        assert abs(ours - ref) < 1e-12


def test_matches_scipy_batch(rng):
    xs = rng.uniform(-50, 10, 500)
    ours = lognormcdf(xs)
    ref = norm.logcdf(xs)
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Extreme tails: stays finite where naive log(normcdf) is -inf
# ---------------------------------------------------------------------------


def test_extreme_negative_still_finite():
    for x in [-40.0, -50.0, -100.0, -200.0]:
        result = lognormcdf(x)
        assert np.isfinite(result)
        # Should be approximately -x²/2
        assert result < -x*x / 2 * 0.99
        assert result > -x*x / 2 * 1.01


def test_extreme_positive_approaches_zero():
    '''log(Φ(x)) → 0 as x → ∞.'''
    assert lognormcdf(10.0) > -1e-20
    assert lognormcdf(20.0) > -1e-80


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_x_zero_is_neg_log_2():
    assert abs(lognormcdf(0.0) - (-np.log(2))) < 1e-14


def test_x_nan_returns_nan():
    result = lognormcdf(np.array([np.nan]))
    assert np.isnan(result[0])


def test_int_promoted_to_float64():
    x = np.array([-1, 0, 1], dtype=np.int64)
    result = lognormcdf(x)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# 4. Batched
# ---------------------------------------------------------------------------


def test_batched(rng):
    x = rng.uniform(-20, 20, 100)
    result = lognormcdf(x)
    for i, xi in enumerate(x):
        assert abs(result[i] - lognormcdf(float(xi))) < 1e-12


# ---------------------------------------------------------------------------
# 5. Warnings suppressed
# ---------------------------------------------------------------------------


def test_no_warnings_across_range():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for x in [-100.0, -40.0, -10.0, 0.0, 10.0, 40.0, 100.0]:
            _ = lognormcdf(x)


# ---------------------------------------------------------------------------
# 6. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    # Restrict to |x| < 6 for the strict-tol check. cupy's log1p rounds
    # tail values to 0 at slightly different points than numpy, causing
    # cosmetic divergence at |x| ~ 8+. Both results are correct to
    # available precision; six sigma covers realistic use.
    x = rng.uniform(-6, 6, 100)
    r_cpu = lognormcdf(x)
    r_gpu = cp.asnumpy(lognormcdf(cp.asarray(x)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, rtol=1e-6)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = lognormcdf(cp.asarray([-3.0, 0.0, 3.0]))
    assert isinstance(result, cp.ndarray)

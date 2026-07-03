'''Test suite for the "stats completion" batch:
rollrange, rollcov, rollbeta, rollmad, rollemastd, rollidio.
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import (
    rollbeta, rollcorr, rollcov, rollemastd, rollidio,
    rollmad, rollmax, rollmin, rollrange, rollstd,
)


# ---------------------------------------------------------------------------
# rollrange
# ---------------------------------------------------------------------------


def test_rollrange_hand_computed():
    x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
    result = rollrange(x, 3)
    expected = [np.nan, np.nan, 3.0, 3.0, 4.0, 8.0, 7.0, 7.0]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_rollrange_matches_max_minus_min(rng):
    x = rng.uniform(-1, 1, size=200)
    np.testing.assert_allclose(rollrange(x, 10), rollmax(x, 10) - rollmin(x, 10), equal_nan=True)


# ---------------------------------------------------------------------------
# rollcov
# ---------------------------------------------------------------------------


def test_rollcov_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=500)
    y = rng.uniform(-1, 1, size=500)
    for w in [5, 20, 50]:
        result = rollcov(x, y, w)
        reference = pd.Series(x).rolling(w, min_periods=w).cov(pd.Series(y)).values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_rollcov_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    y = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=15, replace=False)] = np.nan
    y[rng.choice(300, size=10, replace=False)] = np.nan
    result = rollcov(x, y, 20)
    reference = pd.Series(x).rolling(20, min_periods=20).cov(pd.Series(y)).values
    np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_rollcov_symmetric(rng):
    x = rng.uniform(-1, 1, size=100)
    y = rng.uniform(-1, 1, size=100)
    np.testing.assert_allclose(rollcov(x, y, 10), rollcov(y, x, 10), equal_nan=True)


def test_rollcov_of_x_with_itself_equals_var(rng):
    x = rng.uniform(-1, 1, size=100)
    cov = rollcov(x, x, 10)
    var = rollstd(x, 10) ** 2
    np.testing.assert_allclose(cov, var, atol=1e-12, equal_nan=True)


def test_rollcov_length_mismatch_raises():
    with pytest.raises(ValueError, match='same length'):
        rollcov(np.array([1.0, 2, 3]), np.array([1.0, 2]), 2)


# ---------------------------------------------------------------------------
# rollbeta
# ---------------------------------------------------------------------------


def test_rollbeta_perfect_line():
    '''y = 2x → beta == 2.'''
    x = np.arange(10, dtype=np.float64)
    y = 2.0 * x
    result = rollbeta(x, y, 5)
    np.testing.assert_allclose(result[4:], 2.0, atol=1e-10)


def test_rollbeta_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=500)
    y = 0.5 * x + rng.normal(0, 0.2, size=500)
    for w in [10, 50]:
        result = rollbeta(x, y, w)
        # pandas cov / var == beta
        pcov = pd.Series(x).rolling(w, min_periods=w).cov(pd.Series(y))
        pvar = pd.Series(x).rolling(w, min_periods=w).var()
        reference = (pcov / pvar).values
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_rollbeta_zero_variance_in_x_returns_nan():
    x = np.full(10, 3.0)  # constant x → var(x) = 0 → beta undefined
    y = np.arange(10, dtype=np.float64)
    result = rollbeta(x, y, 5)
    assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# rollmad
# ---------------------------------------------------------------------------


def test_rollmad_hand_computed():
    x = np.array([1.0, 2, 3, 100, 5])
    result = rollmad(x, 5)
    # median = 3, deviations = [2, 1, 0, 97, 2], MAD = median([2,1,0,97,2]) = 2
    assert result[-1] == 2.0


def test_rollmad_robust_to_outlier():
    '''One extreme outlier shouldn't blow up MAD (unlike std).'''
    x_clean = np.random.default_rng(0).uniform(-1, 1, size=99)
    x_outlier = np.concatenate([x_clean, [1000.0]])
    mad_result = rollmad(x_outlier, 100)[-1]
    std_result = rollstd(x_outlier, 100)[-1]
    # MAD should be small (dominated by the clean data)
    assert mad_result < 2.0
    # Std should be blown up by the outlier
    assert std_result > 50.0


def test_rollmad_constant_window_is_zero():
    x = np.full(10, 5.0)
    result = rollmad(x, 5)
    finite = ~np.isnan(result)
    np.testing.assert_allclose(result[finite], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# rollemastd
# ---------------------------------------------------------------------------


def test_rollemastd_matches_pandas_ewm(rng):
    x = rng.uniform(-1, 1, size=500)
    for alpha in [0.1, 0.3, 0.5]:
        result = rollemastd(x, alpha=alpha)
        reference = pd.Series(x).ewm(alpha=alpha, adjust=False).std(bias=False).values
        # First value is NaN in both (undefined for single sample)
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_rollemastd_biased_matches_pandas(rng):
    x = rng.uniform(-1, 1, size=500)
    result = rollemastd(x, alpha=0.3, bias=True)
    reference = pd.Series(x).ewm(alpha=0.3, adjust=False).std(bias=True).values
    np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_rollemastd_neither_span_nor_alpha_raises():
    with pytest.raises(ValueError, match='exactly one'):
        rollemastd(np.array([1.0, 2, 3]))


# ---------------------------------------------------------------------------
# rollidio
# ---------------------------------------------------------------------------


def test_rollidio_perfect_correlation_is_zero(rng):
    '''If y is a linear function of x, all variance is explained → idio ≈ 0.
    The (1 - corr²) term is subject to FP cancellation near corr=1, so we
    allow a modest tolerance rather than strict machine-zero.'''
    x = rng.uniform(-1, 1, size=200)
    y = 2.0 * x + 3.0
    result = rollidio(y, x, 20)
    finite = ~np.isnan(result)
    np.testing.assert_allclose(result[finite], 0.0, atol=1e-6)


def test_rollidio_uncorrelated_equals_std_y(rng):
    '''Uncorrelated → residual std == std(y).'''
    x = rng.uniform(-1, 1, size=500)
    y = rng.uniform(-1, 1, size=500)  # independent
    result = rollidio(y, x, 50)
    std_y = rollstd(y, 50)
    # Not exactly equal because sample corr won't be zero, but should be close
    ratio = result / std_y
    finite = ~np.isnan(ratio)
    # Most windows should have ratio close to 1 (within 20% for finite sample noise)
    assert np.median(ratio[finite]) > 0.85


def test_rollidio_closed_form_matches_manual(rng):
    '''Verify rollidio == sqrt(var(y) * (1 - corr²)).'''
    x = rng.uniform(-1, 1, size=200)
    y = 0.5 * x + rng.normal(0, 0.3, size=200)
    result = rollidio(y, x, 20)
    std_y = rollstd(y, 20)
    corr = rollcorr(x, y, 20)
    manual = std_y * np.sqrt(np.maximum(1 - corr * corr, 0))
    np.testing.assert_allclose(result, manual, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# Common GPU parity tests
# ---------------------------------------------------------------------------


def test_all_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x = rng.uniform(-1, 1, size=200)
    y = rng.uniform(-1, 1, size=200)
    x_g, y_g = cp.asarray(x), cp.asarray(y)
    w = 20

    for name, cpu, gpu in [
        ('rollrange', rollrange(x, w), rollrange(x_g, w)),
        ('rollcov', rollcov(x, y, w), rollcov(x_g, y_g, w)),
        ('rollbeta', rollbeta(x, y, w), rollbeta(x_g, y_g, w)),
        ('rollmad', rollmad(x, w), rollmad(x_g, w)),
        ('rollemastd', rollemastd(x, alpha=0.3), rollemastd(x_g, alpha=0.3)),
        ('rollidio', rollidio(y, x, w), rollidio(y_g, x_g, w)),
    ]:
        np.testing.assert_allclose(cpu, cp.asnumpy(gpu), atol=1e-10, equal_nan=True, err_msg=name)

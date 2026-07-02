'''Test suite for kuant.stats.rollquantile (+ rollmedian, rollpercentile).'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollmedian, rollpercentile, rollquantile


# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------


def test_median_of_progression():
    '''Rolling median of arithmetic progression = middle of window.'''
    x = np.array([1.0, 2, 3, 4, 5])
    result = rollmedian(x, 3)
    np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0], atol=1e-12)


def test_percentile_50_matches_median():
    x = np.arange(20, dtype=np.float64)
    r_p50 = rollpercentile(x, 5, 50)
    r_median = rollmedian(x, 5)
    np.testing.assert_allclose(r_p50, r_median, equal_nan=True)


def test_quantile_0_is_min():
    x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
    result = rollquantile(x, 3, 0.0)
    # Window [3,1,4]: min=1, [1,4,1]: min=1, ...
    expected = [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_quantile_1_is_max():
    x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
    result = rollquantile(x, 3, 1.0)
    expected = [np.nan, np.nan, 4.0, 4.0, 5.0, 9.0, 9.0, 9.0]
    np.testing.assert_allclose(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# Reference match — pandas
# ---------------------------------------------------------------------------


def test_median_matches_pandas(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [3, 10, 50]:
        result = rollmedian(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).median().values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize('q', [0.10, 0.25, 0.75, 0.90])
def test_quantile_matches_pandas(rng, q):
    x = rng.uniform(-1, 1, size=500)
    for w in [10, 30]:
        result = rollquantile(x, w, q)
        reference = pd.Series(x).rolling(w, min_periods=w).quantile(q).values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_quantile_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=20, replace=False)] = np.nan
    for w in [5, 20]:
        result = rollquantile(x, w, 0.5)
        reference = pd.Series(x).rolling(w, min_periods=w).median().values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_window_1_is_identity():
    x = np.array([1.0, 2, 3, 4, 5])
    result = rollmedian(x, 1)
    np.testing.assert_array_equal(result, x)


def test_window_larger_than_length():
    result = rollmedian(np.array([1.0, 2, 3]), 10)
    assert np.all(np.isnan(result))


def test_window_zero_raises():
    with pytest.raises(ValueError, match='must be positive'):
        rollquantile(np.array([1.0, 2, 3]), 0, 0.5)


def test_q_out_of_range_raises():
    with pytest.raises(ValueError, match='q must be in'):
        rollquantile(np.array([1.0, 2, 3]), 2, 1.5)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError, match='p must be in'):
        rollpercentile(np.array([1.0, 2, 3]), 2, 150)


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        rollquantile(np.array([[1.0, 2], [3, 4]]), 2, 0.5)


def test_nan_propagates_in_window():
    x = np.array([1.0, 2, np.nan, 4, 5, 6])
    result = rollmedian(x, 3)
    # Windows overlapping the NaN produce NaN
    assert np.isnan(result[2])  # [1, 2, NaN]
    assert np.isnan(result[3])  # [2, NaN, 4]
    assert np.isnan(result[4])  # [NaN, 4, 5]
    # After NaN falls out
    assert result[5] == 5.0  # [4, 5, 6]


def test_int_input_promoted():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rollmedian(x, 3)
    assert result.dtype == np.float64


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    result = rollmedian(x, 3)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_quantile_monotonic_in_q(rng):
    '''For a fixed window, quantile should be non-decreasing in q.'''
    x = rng.uniform(-1, 1, size=100)
    w = 10
    r_lo = rollquantile(x, w, 0.25)
    r_mid = rollquantile(x, w, 0.5)
    r_hi = rollquantile(x, w, 0.75)
    finite = ~np.isnan(r_mid)
    assert np.all(r_lo[finite] <= r_mid[finite] + 1e-12)
    assert np.all(r_mid[finite] <= r_hi[finite] + 1e-12)


def test_shift_by_constant(rng):
    '''Quantile shifts by a constant when input does.'''
    x = rng.uniform(-1, 1, size=100)
    r1 = rollmedian(x, 10)
    r2 = rollmedian(x + 500, 10)
    finite = ~np.isnan(r1)
    np.testing.assert_allclose((r2 - r1)[finite], 500.0, atol=1e-10)


def test_result_length_equals_input(rng):
    x = rng.uniform(-1, 1, size=87)
    for w in [2, 10, 50]:
        assert rollmedian(x, w).size == 87


# ---------------------------------------------------------------------------
# CPU == GPU
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x_cpu = rng.uniform(-1, 1, size=200)
    x_gpu = cp.asarray(x_cpu)
    for w in [3, 20]:
        r_cpu = rollmedian(x_cpu, w)
        r_gpu = cp.asnumpy(rollmedian(x_gpu, w))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = rollmedian(cp.asarray([1.0, 2, 3, 4, 5]), 3)
    assert isinstance(result, cp.ndarray)

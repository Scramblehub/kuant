"""Test suite for kuant.stats.rollstd.

Validation strategy:
  1. Golden values      — hand-computed
  2. Reference match    — matches pandas .rolling(w, min_periods=w).std(ddof=)
  3. Edge cases         — window=1, w-ddof<=0, all NaN, large-magnitude inputs
  4. Property tests     — non-negative, zero-on-constants, first w-1 NaN
  5. CPU==GPU parity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollstd


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, w, ddof, expected",
    [
        # [1,2,3,4,5] window=3, ddof=1: each window has ssq=2, var=1, std=1
        ([1.0, 2, 3, 4, 5], 3, 1, [np.nan, np.nan, 1.0, 1.0, 1.0]),
        # ddof=0: var=2/3, std=sqrt(2/3)
        ([1.0, 2, 3, 4, 5], 3, 0, [np.nan, np.nan, np.sqrt(2 / 3), np.sqrt(2 / 3), np.sqrt(2 / 3)]),
        # constants → std=0
        ([5.0, 5, 5, 5], 2, 1, [np.nan, 0.0, 0.0, 0.0]),
        # window=n
        ([1.0, 2, 3, 4], 4, 1, [np.nan, np.nan, np.nan, np.std([1, 2, 3, 4], ddof=1)]),
    ],
)
def test_golden_values(x, w, ddof, expected):
    result = rollstd(np.array(x, dtype=np.float64), w, ddof=ddof)
    np.testing.assert_allclose(result, expected, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# 2. Reference match — pandas
# ---------------------------------------------------------------------------


def _pandas_ref(x, w, ddof=1):
    return pd.Series(x).rolling(w, min_periods=w).std(ddof=ddof).values


def test_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=1000)  # small values -> tight tolerance
    for w in [2, 5, 20, 100]:
        for ddof in [0, 1]:
            result = rollstd(x, w, ddof=ddof)
            reference = _pandas_ref(x, w, ddof=ddof)
            np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=500)
    nan_positions = rng.choice(500, size=25, replace=False)
    x[nan_positions] = np.nan
    for w in [3, 10, 30]:
        result = rollstd(x, w)
        reference = _pandas_ref(x, w)
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_matches_pandas_large_magnitude(rng):
    """Shifted cumsum should still match pandas to ~1e-8 on price-like data."""
    x = rng.uniform(3900, 4100, size=500)
    for w in [5, 50]:
        result = rollstd(x, w)
        reference = _pandas_ref(x, w)
        # Wider tolerance for large-magnitude inputs.
        np.testing.assert_allclose(result, reference, atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_window_1_ddof_1_all_nan():
    """w=1, ddof=1 -> w-ddof=0 -> all NaN."""
    result = rollstd(np.array([1.0, 2, 3]), 1, ddof=1)
    assert np.all(np.isnan(result))


def test_window_1_ddof_0_all_zeros():
    """w=1, ddof=0 -> variance of single value = 0."""
    result = rollstd(np.array([1.0, 2, 3, 4]), 1, ddof=0)
    np.testing.assert_allclose(result, [0, 0, 0, 0], atol=1e-15)


def test_window_larger_than_length_all_nan():
    result = rollstd(np.array([1.0, 2, 3]), 10)
    assert np.all(np.isnan(result))


def test_window_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        rollstd(np.array([1.0, 2, 3]), 0)


def test_ddof_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        rollstd(np.array([1.0, 2, 3]), 2, ddof=-1)


def test_2d_input_raises():
    with pytest.raises(ValueError, match="1D"):
        rollstd(np.array([[1.0, 2], [3, 4]]), 2)


def test_all_nan_input():
    x = np.full(10, np.nan)
    result = rollstd(x, 3)
    assert np.all(np.isnan(result))


def test_first_element_nan_still_works():
    """If x[0] is NaN, shift should fall back to 0 without cascading NaN."""
    x = np.array([np.nan, 1.0, 2, 3, 4, 5])
    result = rollstd(x, 3)
    reference = _pandas_ref(x, 3)
    np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_isolated_nan_recovery():
    """A single NaN poisons its overlapping windows, then computation resumes."""
    x = np.arange(10, dtype=np.float64)
    x[5] = np.nan
    result = rollstd(x, 3)
    reference = _pandas_ref(x, 3)
    np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_int_input_promoted_to_float64():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rollstd(x, 3)
    assert result.dtype == np.float64


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    result = rollstd(x, 3)
    assert result.dtype == np.float32


def test_python_list_input():
    result = rollstd([1.0, 2, 3, 4, 5], 3)
    np.testing.assert_allclose(result[2:], [1.0, 1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_result_nonnegative(rng):
    x = rng.uniform(-1, 1, size=200)
    result = rollstd(x, 10)
    # NaN OR non-negative
    finite = ~np.isnan(result)
    assert np.all(result[finite] >= 0.0)


def test_zero_on_constants():
    x = np.full(20, 7.0)
    result = rollstd(x, 5)
    finite = ~np.isnan(result)
    np.testing.assert_allclose(result[finite], 0.0, atol=1e-12)


def test_first_w_minus_1_always_nan(rng):
    x = rng.uniform(-1, 1, size=100)
    for w in [2, 5, 20]:
        result = rollstd(x, w)
        assert np.all(np.isnan(result[: w - 1]))


def test_result_length_equals_input(rng):
    x = rng.uniform(-1, 1, size=87)
    for w in [2, 10, 50]:
        assert rollstd(x, w).size == 87


def test_shift_invariance(rng):
    """Adding a constant to x should not change rollstd."""
    x = rng.uniform(-1, 1, size=100)
    result1 = rollstd(x, 10)
    result2 = rollstd(x + 1000, 10)
    np.testing.assert_allclose(result1, result2, atol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# 5. CPU == GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    x_cpu = rng.uniform(-1, 1, size=1000)
    x_gpu = cp.asarray(x_cpu)
    for w in [3, 10, 50]:
        r_cpu = rollstd(x_cpu, w)
        r_gpu = cp.asnumpy(rollstd(x_gpu, w))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp

    result = rollstd(cp.asarray([1.0, 2, 3, 4, 5]), 3)
    assert isinstance(result, cp.ndarray)


def test_gpu_with_nans(skip_no_gpu, rng):
    import cupy as cp

    x_cpu = rng.uniform(-1, 1, size=200)
    x_cpu[[10, 50, 100, 150]] = np.nan
    x_gpu = cp.asarray(x_cpu)
    r_cpu = rollstd(x_cpu, 5)
    r_gpu = cp.asnumpy(rollstd(x_gpu, 5))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)

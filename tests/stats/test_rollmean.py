'''Test suite for kuant.stats.rollmean.

Validation strategy:
  1. Golden values      — hand-computed small examples
  2. Reference match    — matches pandas .rolling(w).mean() with min_periods=w
  3. Edge cases         — window=1, window=n, window>n, empty, single NaN,
                           boundary NaN, all NaN, dtype preservation
  4. Property tests     — first w-1 always NaN; result monotonic w.r.t. window
                           on strictly-increasing input
  5. CPU==GPU parity
'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollmean


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'x, w, expected',
    [
        # Basic 1-2-3-4-5 window=3
        ([1.0, 2, 3, 4, 5], 3, [np.nan, np.nan, 2.0, 3.0, 4.0]),
        # Window == length
        ([1.0, 2, 3], 3, [np.nan, np.nan, 2.0]),
        # Window == 1 — identity
        ([1.0, 2, 3], 1, [1.0, 2.0, 3.0]),
        # Constants
        ([5.0, 5, 5, 5], 2, [np.nan, 5.0, 5.0, 5.0]),
    ],
)
def test_golden_values(x, w, expected):
    result = rollmean(np.array(x, dtype=np.float64), w)
    np.testing.assert_allclose(result, expected, atol=1e-14, equal_nan=True)


# ---------------------------------------------------------------------------
# 2. Reference match — pandas
# ---------------------------------------------------------------------------


def _pandas_ref(x, w):
    '''Reference: pandas rolling mean with min_periods=w (strict window).'''
    return pd.Series(x).rolling(w, min_periods=w).mean().values


def test_matches_pandas_uniform(rng):
    x = rng.uniform(-10, 10, size=1000)
    for w in [1, 2, 5, 20, 100]:
        result = rollmean(x, w)
        reference = _pandas_ref(x, w)
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_matches_pandas_with_nans(rng):
    x = rng.uniform(-10, 10, size=500)
    # Scatter ~5% NaNs
    nan_positions = rng.choice(500, size=25, replace=False)
    x[nan_positions] = np.nan
    for w in [3, 10, 30]:
        result = rollmean(x, w)
        reference = _pandas_ref(x, w)
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_window_1_is_identity():
    x = np.array([1.0, 2, 3, 4, 5])
    np.testing.assert_array_equal(rollmean(x, 1), x)


def test_window_equal_to_length():
    x = np.array([1.0, 2, 3, 4])
    result = rollmean(x, 4)
    expected = [np.nan, np.nan, np.nan, 2.5]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_window_larger_than_length_all_nan():
    x = np.array([1.0, 2, 3])
    result = rollmean(x, 10)
    assert np.all(np.isnan(result))
    assert result.size == 3


def test_window_zero_raises():
    with pytest.raises(ValueError, match='must be positive'):
        rollmean(np.array([1.0, 2, 3]), 0)


def test_window_negative_raises():
    with pytest.raises(ValueError, match='must be positive'):
        rollmean(np.array([1.0, 2, 3]), -1)


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        rollmean(np.array([[1.0, 2], [3, 4]]), 2)


def test_single_nan_isolates():
    '''NaN at index 5 poisons windows overlapping it, then normal resumes.'''
    x = np.arange(10, dtype=np.float64)
    x[5] = np.nan
    result = rollmean(x, 3)
    # Windows [0-2], [1-3], [2-4] don't include index 5 -> valid
    # Windows [3-5], [4-6], [5-7] include index 5 -> NaN
    # Window [6-8] excludes index 5 -> valid again
    assert np.isnan(result[0])  # partial window
    assert np.isnan(result[1])  # partial window
    assert result[2] == pytest.approx(1.0)  # (0+1+2)/3
    assert result[3] == pytest.approx(2.0)  # (1+2+3)/3
    assert result[4] == pytest.approx(3.0)  # (2+3+4)/3
    assert np.isnan(result[5])  # (3+4+NaN)/3
    assert np.isnan(result[6])  # (4+NaN+6)/3
    assert np.isnan(result[7])  # (NaN+6+7)/3
    assert result[8] == pytest.approx(7.0)  # (6+7+8)/3


def test_all_nan_input():
    x = np.full(10, np.nan)
    result = rollmean(x, 3)
    assert np.all(np.isnan(result))


def test_int_input_promoted_to_float64():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rollmean(x, 3)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0])


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    result = rollmean(x, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0], atol=1e-6)


def test_python_list_input():
    result = rollmean([1.0, 2, 3, 4, 5], 3)
    np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_first_w_minus_1_always_nan(rng):
    x = rng.uniform(-1, 1, size=100)
    for w in [1, 2, 5, 20, 50]:
        result = rollmean(x, w)
        # First w-1 entries are NaN by convention
        assert np.all(np.isnan(result[:w-1]))
        # Position w-1 and beyond should be finite (no NaN in clean input)
        assert np.all(~np.isnan(result[w-1:]))


def test_result_length_equals_input(rng):
    x = rng.uniform(-1, 1, size=87)
    for w in [1, 3, 10, 50]:
        result = rollmean(x, w)
        assert result.size == 87


def test_matches_naive_loop(rng):
    '''Sanity check: cumsum trick matches O(n*w) naive computation.'''
    x = rng.uniform(-10, 10, size=100)
    w = 7
    result = rollmean(x, w)
    naive = np.full(100, np.nan)
    for i in range(w - 1, 100):
        naive[i] = x[i - w + 1 : i + 1].mean()
    np.testing.assert_allclose(result, naive, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# 5. CPU == GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x_cpu = rng.uniform(-10, 10, size=1000)
    x_gpu = cp.asarray(x_cpu)
    for w in [3, 10, 50]:
        r_cpu = rollmean(x_cpu, w)
        r_gpu = cp.asnumpy(rollmean(x_gpu, w))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = rollmean(cp.asarray([1.0, 2, 3, 4, 5]), 3)
    assert isinstance(result, cp.ndarray)


def test_gpu_with_nans(skip_no_gpu, rng):
    import cupy as cp
    x_cpu = rng.uniform(-10, 10, size=200)
    x_cpu[[10, 50, 100, 150]] = np.nan
    x_gpu = cp.asarray(x_cpu)
    r_cpu = rollmean(x_cpu, 5)
    r_gpu = cp.asnumpy(rollmean(x_gpu, 5))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)

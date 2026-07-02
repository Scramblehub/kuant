'''Test suite for kuant.stats.rollmin and rollmax.'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollmax, rollmin


# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------


def test_rollmin_hand_computed():
    x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
    result = rollmin(x, 3)
    expected = [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_rollmax_hand_computed():
    x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
    result = rollmax(x, 3)
    expected = [np.nan, np.nan, 4.0, 4.0, 5.0, 9.0, 9.0, 9.0]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_window_1_is_identity():
    x = np.array([3.0, 1, 4, 1, 5])
    np.testing.assert_array_equal(rollmin(x, 1), x)
    np.testing.assert_array_equal(rollmax(x, 1), x)


# ---------------------------------------------------------------------------
# Reference match — pandas
# ---------------------------------------------------------------------------


def test_rollmin_matches_pandas(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [3, 10, 50]:
        result = rollmin(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).min().values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_rollmax_matches_pandas(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [3, 10, 50]:
        result = rollmax(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).max().values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=15, replace=False)] = np.nan
    for w in [5, 20]:
        r_min = rollmin(x, w)
        r_max = rollmax(x, w)
        ref_min = pd.Series(x).rolling(w, min_periods=w).min().values
        ref_max = pd.Series(x).rolling(w, min_periods=w).max().values
        np.testing.assert_allclose(r_min, ref_min, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(r_max, ref_max, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_window_larger_than_length():
    x = np.array([1.0, 2, 3])
    assert np.all(np.isnan(rollmin(x, 10)))
    assert np.all(np.isnan(rollmax(x, 10)))


def test_window_zero_raises():
    with pytest.raises(ValueError, match='must be positive'):
        rollmin(np.array([1.0, 2]), 0)
    with pytest.raises(ValueError, match='must be positive'):
        rollmax(np.array([1.0, 2]), 0)


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        rollmin(np.array([[1.0, 2], [3, 4]]), 2)


def test_nan_in_window_propagates():
    x = np.array([1.0, 2, np.nan, 4, 5, 6])
    r_min = rollmin(x, 3)
    r_max = rollmax(x, 3)
    # Windows overlapping the NaN → NaN
    for i in [2, 3, 4]:
        assert np.isnan(r_min[i])
        assert np.isnan(r_max[i])
    # After NaN falls out
    assert r_min[5] == 4.0
    assert r_max[5] == 6.0


def test_int_input_promoted():
    x = np.array([3, 1, 4, 1, 5], dtype=np.int64)
    result = rollmin(x, 3)
    assert result.dtype == np.float64


def test_dtype_preserved_float32():
    x = np.array([3.0, 1, 4, 1, 5], dtype=np.float32)
    assert rollmin(x, 3).dtype == np.float32
    assert rollmax(x, 3).dtype == np.float32


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_min_leq_max(rng):
    x = rng.uniform(-1, 1, size=100)
    r_min = rollmin(x, 10)
    r_max = rollmax(x, 10)
    finite = ~np.isnan(r_min)
    assert np.all(r_min[finite] <= r_max[finite])


def test_min_neg_max_neg(rng):
    '''min(-x) == -max(x).'''
    x = rng.uniform(-1, 1, size=100)
    r_min_neg = rollmin(-x, 10)
    r_max = rollmax(x, 10)
    np.testing.assert_allclose(r_min_neg, -r_max, atol=1e-14, equal_nan=True)


def test_first_w_minus_1_nan():
    x = np.arange(20, dtype=np.float64)
    for w in [3, 10]:
        assert np.all(np.isnan(rollmin(x, w)[:w-1]))
        assert np.all(np.isnan(rollmax(x, w)[:w-1]))


def test_monotonically_increasing_input():
    '''For a strictly increasing input, rollmax[i] = x[i], rollmin[i] = x[i-w+1].'''
    x = np.arange(20, dtype=np.float64)
    w = 5
    r_min = rollmin(x, w)
    r_max = rollmax(x, w)
    for i in range(w-1, 20):
        assert r_max[i] == x[i]
        assert r_min[i] == x[i - w + 1]


# ---------------------------------------------------------------------------
# CPU == GPU
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x_cpu = rng.uniform(-1, 1, size=200)
    x_gpu = cp.asarray(x_cpu)
    for w in [5, 30]:
        r_min_cpu = rollmin(x_cpu, w)
        r_min_gpu = cp.asnumpy(rollmin(x_gpu, w))
        np.testing.assert_allclose(r_min_cpu, r_min_gpu, atol=1e-12, equal_nan=True)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    assert isinstance(rollmin(cp.asarray([1.0, 2, 3]), 2), cp.ndarray)
    assert isinstance(rollmax(cp.asarray([1.0, 2, 3]), 2), cp.ndarray)

"""Test suite for kuant.stats.rollargmin and rollargmax."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import rollargmax, rollargmin


def test_rollargmax_hand_computed():
    x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
    result = rollargmax(x, 3)
    # Windows: [3,1,4] argmax=2; [1,4,1] argmax=1; [4,1,5] argmax=2;
    #          [1,5,9] argmax=2; [5,9,2] argmax=1; [9,2,6] argmax=0
    expected = [np.nan, np.nan, 2.0, 1.0, 2.0, 2.0, 1.0, 0.0]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_rollargmin_hand_computed():
    x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
    result = rollargmin(x, 3)
    # Windows: [3,1,4]=1; [1,4,1]=0; [4,1,5]=1; [1,5,9]=0; [5,9,2]=2; [9,2,6]=1
    expected = [np.nan, np.nan, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_ascending_argmax_is_last():
    x = np.arange(10, dtype=np.float64)
    result = rollargmax(x, 3)
    # Every window's max is the last element
    np.testing.assert_allclose(result[2:], [2.0] * 8)


def test_ascending_argmin_is_first():
    x = np.arange(10, dtype=np.float64)
    result = rollargmin(x, 3)
    np.testing.assert_allclose(result[2:], [0.0] * 8)


def test_window_1_all_zeros():
    x = np.array([3.0, 1, 4, 1, 5])
    np.testing.assert_array_equal(rollargmax(x, 1), np.zeros(5))
    np.testing.assert_array_equal(rollargmin(x, 1), np.zeros(5))


def test_ties_return_first():
    """numpy convention: argmax of tie returns first occurrence."""
    x = np.array([1.0, 5, 5, 5, 5])
    result = rollargmax(x, 4)
    # Window [1,5,5,5]: first max at index 1; window [5,5,5,5]: first max at 0
    assert result[3] == 1.0
    assert result[4] == 0.0


def test_nan_in_window_propagates():
    x = np.array([3.0, 1, np.nan, 4, 5, 6])
    r_min = rollargmin(x, 3)
    r_max = rollargmax(x, 3)
    for i in [2, 3, 4]:
        assert np.isnan(r_min[i])
        assert np.isnan(r_max[i])
    # After NaN falls out
    assert r_min[5] == 0.0  # [4,5,6] min at 0
    assert r_max[5] == 2.0  # max at 2


def test_window_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        rollargmax(np.array([1.0]), 0)
    with pytest.raises(ValueError, match="must be positive"):
        rollargmin(np.array([1.0]), 0)


def test_window_larger_than_length():
    assert np.all(np.isnan(rollargmax(np.array([1.0, 2, 3]), 10)))


def test_2d_input_raises():
    with pytest.raises(ValueError, match="1D"):
        rollargmax(np.array([[1.0, 2], [3, 4]]), 2)


def test_dtype_preserved_float32():
    x = np.array([3.0, 1, 4, 1, 5], dtype=np.float32)
    assert rollargmax(x, 3).dtype == np.float32


def test_bounded_by_0_and_w_minus_1(rng):
    x = rng.uniform(-1, 1, size=100)
    w = 10
    result = rollargmax(x, w)
    finite = ~np.isnan(result)
    assert np.all(result[finite] >= 0)
    assert np.all(result[finite] <= w - 1)


def test_argmax_neg_equals_argmin(rng):
    """argmax(-x) == argmin(x)."""
    x = rng.uniform(-1, 1, size=100)
    np.testing.assert_array_equal(rollargmax(-x, 10), rollargmin(x, 10))


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    x = rng.uniform(-1, 1, size=200)
    for w in [3, 20]:
        r_cpu = rollargmax(x, w)
        r_gpu = cp.asnumpy(rollargmax(cp.asarray(x), w))
        np.testing.assert_allclose(r_cpu, r_gpu, equal_nan=True)

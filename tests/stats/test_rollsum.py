'''Test suite for kuant.stats.rollsum.'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollmean, rollsum


def test_golden():
    x = np.array([1.0, 2, 3, 4, 5])
    result = rollsum(x, 3)
    np.testing.assert_allclose(result, [np.nan, np.nan, 6.0, 9.0, 12.0], equal_nan=True)


def test_matches_pandas(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [3, 10, 50]:
        result = rollsum(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).sum().values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=15, replace=False)] = np.nan
    result = rollsum(x, 10)
    reference = pd.Series(x).rolling(10, min_periods=10).sum().values
    np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_equals_rollmean_times_w():
    x = np.arange(20, dtype=np.float64)
    for w in [3, 5]:
        np.testing.assert_allclose(rollsum(x, w), rollmean(x, w) * w, equal_nan=True)


def test_window_zero_raises():
    with pytest.raises(ValueError, match='must be positive'):
        rollsum(np.array([1.0, 2]), 0)


def test_window_larger_than_length():
    assert np.all(np.isnan(rollsum(np.array([1.0, 2, 3]), 10)))


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        rollsum(np.array([[1.0, 2], [3, 4]]), 2)


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    assert rollsum(x, 3).dtype == np.float32


def test_int_input_promoted():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert rollsum(x, 3).dtype == np.float64


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x = rng.uniform(-1, 1, size=200)
    np.testing.assert_allclose(
        rollsum(x, 10), cp.asnumpy(rollsum(cp.asarray(x), 10)),
        atol=1e-12, equal_nan=True,
    )

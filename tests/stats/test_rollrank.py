'''Test suite for kuant.stats.rollrank.'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollrank


# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------


def test_unique_values_ordinal():
    '''Strictly ascending: last element is always max → rank == w.'''
    x = np.arange(10, dtype=np.float64)
    result = rollrank(x, 3)
    np.testing.assert_allclose(result[2:], [3.0] * 8, atol=1e-12)


def test_unique_values_descending():
    '''Strictly descending: last element is always min → rank == 1.'''
    x = np.arange(10, dtype=np.float64)[::-1].copy()
    result = rollrank(x, 3)
    np.testing.assert_allclose(result[2:], [1.0] * 8, atol=1e-12)


def test_average_rank_for_ties():
    '''Window [1, 4, 1] with last=1: less=0, equal=2, rank=(0+3/2)=1.5.'''
    x = np.array([3.0, 1, 4, 1, 5])
    result = rollrank(x, 3)
    # Windows: [3,1,4] last=4 rank=3; [1,4,1] last=1 rank=1.5; [4,1,5] last=5 rank=3.
    np.testing.assert_allclose(result, [np.nan, np.nan, 3.0, 1.5, 3.0], equal_nan=True)


def test_pct_normalization():
    x = np.array([3.0, 1, 4, 1, 5])
    result = rollrank(x, 3, pct=True)
    np.testing.assert_allclose(result, [np.nan, np.nan, 1.0, 0.5, 1.0], equal_nan=True)


def test_all_equal_window():
    '''All-equal window: rank = (w + 1) / 2 (average of 1..w).'''
    x = np.full(6, 7.0)
    result = rollrank(x, 3)
    # equal=3, less=0, rank = 0 + (3+1)/2 = 2.0
    np.testing.assert_allclose(result[2:], [2.0] * 4, atol=1e-12)


# ---------------------------------------------------------------------------
# Reference match — pandas
# ---------------------------------------------------------------------------


def test_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [3, 10, 50]:
        result = rollrank(x, w)
        # Note: pandas returns rank/w for pct=True; we default to raw rank.
        reference = pd.Series(x).rolling(w, min_periods=w).rank().values
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_matches_pandas_pct(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [5, 20]:
        result = rollrank(x, w, pct=True)
        reference = pd.Series(x).rolling(w, min_periods=w).rank(pct=True).values
        np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_matches_pandas_with_ties(rng):
    '''Discrete input space forces frequent ties.'''
    x = rng.integers(0, 5, size=300).astype(np.float64)
    for w in [4, 10]:
        result = rollrank(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).rank().values
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=15, replace=False)] = np.nan
    for w in [5, 20]:
        result = rollrank(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).rank().values
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_window_1_all_ones():
    '''Rank of single element in its 1-window is always 1.'''
    x = np.array([3.0, 1, 4, 1, 5])
    result = rollrank(x, 1)
    np.testing.assert_array_equal(result, np.ones(5))


def test_window_1_pct_all_ones():
    x = np.array([3.0, 1, 4, 1, 5])
    result = rollrank(x, 1, pct=True)
    np.testing.assert_array_equal(result, np.ones(5))


def test_window_larger_than_length():
    x = np.array([1.0, 2, 3])
    assert np.all(np.isnan(rollrank(x, 10)))


def test_window_zero_raises():
    with pytest.raises(ValueError, match='must be positive'):
        rollrank(np.array([1.0, 2]), 0)


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        rollrank(np.array([[1.0, 2], [3, 4]]), 2)


def test_nan_in_window_propagates():
    x = np.array([1.0, 2, np.nan, 4, 5, 6])
    result = rollrank(x, 3)
    reference = pd.Series(x).rolling(3, min_periods=3).rank().values
    np.testing.assert_allclose(result, reference, atol=1e-12, equal_nan=True)


def test_int_input_promoted():
    x = np.array([3, 1, 4, 1, 5], dtype=np.int64)
    result = rollrank(x, 3)
    assert result.dtype == np.float64


def test_dtype_preserved_float32():
    x = np.array([3.0, 1, 4, 1, 5], dtype=np.float32)
    result = rollrank(x, 3)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_rank_bounded_by_1_and_w(rng):
    x = rng.uniform(-1, 1, size=200)
    w = 10
    result = rollrank(x, w)
    finite = ~np.isnan(result)
    assert np.all(result[finite] >= 1.0 - 1e-12)
    assert np.all(result[finite] <= w + 1e-12)


def test_pct_bounded_by_0_and_1(rng):
    x = rng.uniform(-1, 1, size=200)
    result = rollrank(x, 10, pct=True)
    finite = ~np.isnan(result)
    assert np.all(result[finite] > 0.0 - 1e-12)
    assert np.all(result[finite] <= 1.0 + 1e-12)


def test_shift_invariance(rng):
    '''Rank is invariant under a strictly-monotonic transform, e.g. a shift.'''
    x = rng.uniform(-1, 1, size=100)
    r1 = rollrank(x, 10)
    r2 = rollrank(x + 500, 10)
    np.testing.assert_allclose(r1, r2, atol=1e-12, equal_nan=True)


def test_scale_invariance_positive(rng):
    '''Rank is invariant under positive scaling.'''
    x = rng.uniform(-1, 1, size=100)
    r1 = rollrank(x, 10)
    r2 = rollrank(x * 3.7, 10)
    np.testing.assert_allclose(r1, r2, atol=1e-12, equal_nan=True)


def test_negation_reverses_rank(rng):
    '''rollrank(-x, w) == w + 1 - rollrank(x, w) for the raw ranks.'''
    x = rng.uniform(-1, 1, size=100)
    w = 10
    r1 = rollrank(x, w)
    r2 = rollrank(-x, w)
    finite = ~np.isnan(r1)
    np.testing.assert_allclose(r1[finite] + r2[finite], w + 1, atol=1e-12)


# ---------------------------------------------------------------------------
# CPU == GPU
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x_cpu = rng.uniform(-1, 1, size=200)
    x_gpu = cp.asarray(x_cpu)
    for w in [5, 30]:
        r_cpu = rollrank(x_cpu, w)
        r_gpu = cp.asnumpy(rollrank(x_gpu, w))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = rollrank(cp.asarray([3.0, 1, 4, 1, 5]), 3)
    assert isinstance(result, cp.ndarray)

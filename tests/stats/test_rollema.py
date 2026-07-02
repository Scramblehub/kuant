'''Test suite for kuant.stats.rollema.'''
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollema


def test_first_value_equals_input():
    x = np.array([5.0, 2, 3, 4, 5])
    result = rollema(x, alpha=0.5)
    assert result[0] == 5.0


def test_constant_input_stays_constant():
    x = np.full(10, 7.0)
    result = rollema(x, alpha=0.3)
    np.testing.assert_allclose(result, 7.0, atol=1e-12)


def test_recursion_manual_verification():
    '''ema[1] = alpha*x[1] + (1-alpha)*ema[0].'''
    x = np.array([1.0, 2, 3, 4, 5])
    alpha = 0.5
    result = rollema(x, alpha=alpha)
    expected = np.zeros(5)
    expected[0] = 1.0
    for i in range(1, 5):
        expected[i] = alpha * x[i] + (1 - alpha) * expected[i-1]
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_matches_pandas_ewm(rng):
    '''Matches pandas ewm(alpha=..., adjust=False).mean().'''
    x = rng.uniform(-1, 1, size=500)
    for alpha in [0.1, 0.3, 0.5, 0.9]:
        result = rollema(x, alpha=alpha)
        reference = pd.Series(x).ewm(alpha=alpha, adjust=False).mean().values
        np.testing.assert_allclose(result, reference, atol=1e-10)


def test_span_matches_pandas(rng):
    x = rng.uniform(-1, 1, size=500)
    for span in [5, 20, 100]:
        result = rollema(x, span=span)
        reference = pd.Series(x).ewm(span=span, adjust=False).mean().values
        np.testing.assert_allclose(result, reference, atol=1e-10)


def test_span_1_equivalent_alpha_1():
    '''span=1 → alpha = 2/2 = 1 → ema[i] = x[i] (no smoothing).'''
    x = np.array([1.0, 2, 3, 4, 5])
    result = rollema(x, span=1)
    np.testing.assert_allclose(result, x, atol=1e-12)


def test_neither_span_nor_alpha_raises():
    with pytest.raises(ValueError, match='exactly one'):
        rollema(np.array([1.0, 2, 3]))


def test_both_span_and_alpha_raises():
    with pytest.raises(ValueError, match='exactly one'):
        rollema(np.array([1.0, 2, 3]), span=5, alpha=0.5)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match='alpha must be'):
        rollema(np.array([1.0, 2, 3]), alpha=1.5)
    with pytest.raises(ValueError, match='alpha must be'):
        rollema(np.array([1.0, 2, 3]), alpha=0.0)


def test_span_less_than_1_raises():
    with pytest.raises(ValueError, match='span must be'):
        rollema(np.array([1.0, 2, 3]), span=0.5)


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        rollema(np.array([[1.0, 2], [3, 4]]), alpha=0.5)


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    result = rollema(x, alpha=0.5)
    assert result.dtype == np.float32


def test_int_input_promoted():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rollema(x, alpha=0.5)
    assert result.dtype == np.float64


def test_empty_input():
    x = np.array([], dtype=np.float64)
    result = rollema(x, alpha=0.5)
    assert result.size == 0


def test_low_alpha_high_smoothing():
    '''alpha=0.01 → ema barely responds to input step.'''
    x = np.concatenate([np.zeros(50), np.ones(50)])
    result = rollema(x, alpha=0.01)
    # After 50 steps of ones with alpha=0.01, ema should still be much less than 1
    assert result[99] < 0.5


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x = rng.uniform(-1, 1, size=200)
    r_cpu = rollema(x, alpha=0.3)
    r_gpu = cp.asnumpy(rollema(cp.asarray(x), alpha=0.3))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = rollema(cp.asarray([1.0, 2, 3, 4, 5]), alpha=0.5)
    assert isinstance(result, cp.ndarray)

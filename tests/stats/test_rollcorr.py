"""Test suite for kuant.stats.rollcorr."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollcorr


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


def test_perfect_positive_correlation():
    """y = 2x for all i → corr = 1 everywhere."""
    x = np.array([1.0, 2, 3, 4, 5])
    y = 2 * x
    result = rollcorr(x, y, 3)
    np.testing.assert_allclose(result[2:], [1.0, 1.0, 1.0], atol=1e-12)


def test_perfect_negative_correlation():
    x = np.array([1.0, 2, 3, 4, 5])
    y = -3 * x + 10
    result = rollcorr(x, y, 3)
    np.testing.assert_allclose(result[2:], [-1.0, -1.0, -1.0], atol=1e-12)


def test_uncorrelated_series():
    """Two orthogonal patterns → corr near 0 over the full-length window."""
    x = np.array([1.0, -1, 1, -1])
    y = np.array([1.0, 1, -1, -1])
    result = rollcorr(x, y, 4)
    assert abs(result[3]) < 1e-12


# ---------------------------------------------------------------------------
# 2. Reference match — pandas
# ---------------------------------------------------------------------------


def _pandas_ref(x, y, w):
    return pd.Series(x).rolling(w, min_periods=w).corr(pd.Series(y)).values


def test_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=500)
    y = rng.uniform(-1, 1, size=500)
    for w in [3, 10, 50]:
        result = rollcorr(x, y, w)
        reference = _pandas_ref(x, y, w)
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_matches_pandas_correlated_signal(rng):
    """y = x + small noise → non-trivial correlation the reference must match."""
    x = rng.uniform(-1, 1, size=500)
    y = 0.5 * x + rng.normal(0, 0.1, size=500)
    for w in [5, 30]:
        result = rollcorr(x, y, w)
        reference = _pandas_ref(x, y, w)
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    y = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=15, replace=False)] = np.nan
    y[rng.choice(300, size=10, replace=False)] = np.nan
    for w in [5, 20]:
        result = rollcorr(x, y, w)
        reference = _pandas_ref(x, y, w)
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_matches_pandas_large_magnitude(rng):
    x = rng.uniform(3900, 4100, size=300)
    y = rng.uniform(3900, 4100, size=300)
    for w in [10, 30]:
        result = rollcorr(x, y, w)
        reference = _pandas_ref(x, y, w)
        np.testing.assert_allclose(result, reference, atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_result_in_range(rng):
    x = rng.uniform(-1, 1, size=200)
    y = rng.uniform(-1, 1, size=200)
    result = rollcorr(x, y, 10)
    finite = ~np.isnan(result)
    assert np.all(result[finite] >= -1.0 - 1e-12)
    assert np.all(result[finite] <= 1.0 + 1e-12)


def test_zero_variance_returns_nan():
    """Constant y → rollstd_y = 0 → corr undefined → NaN."""
    x = np.array([1.0, 2, 3, 4, 5])
    y = np.full(5, 3.0)
    result = rollcorr(x, y, 3)
    assert np.all(np.isnan(result))


def test_window_1_all_nan():
    """Correlation of single points is undefined."""
    x = np.array([1.0, 2, 3, 4])
    y = np.array([2.0, 3, 4, 5])
    result = rollcorr(x, y, 1)
    assert np.all(np.isnan(result))


def test_window_larger_than_length():
    x = np.array([1.0, 2, 3])
    y = np.array([2.0, 3, 4])
    result = rollcorr(x, y, 10)
    assert np.all(np.isnan(result))


def test_window_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        rollcorr(np.array([1.0, 2]), np.array([3.0, 4]), 0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        rollcorr(np.array([1.0, 2, 3]), np.array([1.0, 2]), 2)


def test_2d_input_raises():
    with pytest.raises(ValueError, match="1D"):
        rollcorr(np.array([[1.0, 2], [3, 4]]), np.array([[1.0, 2], [3, 4]]), 2)


def test_nan_in_x_only():
    """NaN in x poisons the union → windows overlapping NaN produce NaN."""
    x = np.array([1.0, 2, np.nan, 4, 5, 6])
    y = np.array([1.0, 2, 3, 4, 5, 6])
    result = rollcorr(x, y, 3)
    reference = _pandas_ref(x, y, 3)
    np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_nan_in_y_only():
    x = np.array([1.0, 2, 3, 4, 5, 6])
    y = np.array([1.0, 2, 3, np.nan, 5, 6])
    result = rollcorr(x, y, 3)
    reference = _pandas_ref(x, y, 3)
    np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_int_inputs_promoted():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    y = np.array([2, 4, 6, 8, 10], dtype=np.int64)
    result = rollcorr(x, y, 3)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result[2:], [1.0, 1.0, 1.0], atol=1e-12)


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    y = np.array([2.0, 4, 6, 8, 10], dtype=np.float32)
    result = rollcorr(x, y, 3)
    assert result.dtype == np.float32


def test_python_list_input():
    result = rollcorr([1.0, 2, 3, 4, 5], [2.0, 4, 6, 8, 10], 3)
    np.testing.assert_allclose(result[2:], [1.0, 1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_symmetry(rng):
    """corr(x, y) == corr(y, x)."""
    x = rng.uniform(-1, 1, size=100)
    y = rng.uniform(-1, 1, size=100)
    r1 = rollcorr(x, y, 10)
    r2 = rollcorr(y, x, 10)
    np.testing.assert_allclose(r1, r2, atol=1e-12, equal_nan=True)


def test_shift_invariance(rng):
    """corr(x + a, y + b) == corr(x, y) for any a, b."""
    x = rng.uniform(-1, 1, size=100)
    y = rng.uniform(-1, 1, size=100)
    r1 = rollcorr(x, y, 10)
    r2 = rollcorr(x + 1000, y - 500, 10)
    np.testing.assert_allclose(r1, r2, atol=1e-8, equal_nan=True)


def test_scale_invariance_positive(rng):
    """corr(a*x, b*y) == corr(x, y) for a, b > 0."""
    x = rng.uniform(0.1, 1.0, size=100)
    y = rng.uniform(0.1, 1.0, size=100)
    r1 = rollcorr(x, y, 10)
    r2 = rollcorr(3.0 * x, 7.0 * y, 10)
    np.testing.assert_allclose(r1, r2, atol=1e-10, equal_nan=True)


def test_scale_sign_flip(rng):
    """corr(x, -y) == -corr(x, y)."""
    x = rng.uniform(-1, 1, size=100)
    y = rng.uniform(-1, 1, size=100)
    r1 = rollcorr(x, y, 10)
    r2 = rollcorr(x, -y, 10)
    np.testing.assert_allclose(r1, -r2, atol=1e-10, equal_nan=True)


def test_first_w_minus_1_nan():
    x = np.arange(20, dtype=np.float64)
    y = x * 2
    for w in [2, 5, 10]:
        result = rollcorr(x, y, w)
        assert np.all(np.isnan(result[: w - 1]))


# ---------------------------------------------------------------------------
# 5. CPU == GPU
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    x_cpu = rng.uniform(-1, 1, size=500)
    y_cpu = rng.uniform(-1, 1, size=500)
    for w in [5, 20, 50]:
        r_cpu = rollcorr(x_cpu, y_cpu, w)
        r_gpu = cp.asnumpy(rollcorr(cp.asarray(x_cpu), cp.asarray(y_cpu), w))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp

    result = rollcorr(cp.asarray([1.0, 2, 3, 4, 5]), cp.asarray([2.0, 4, 6, 8, 10]), 3)
    assert isinstance(result, cp.ndarray)

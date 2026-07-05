"""Test suite for kuant.stats.rollskew and rollkurt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import rollkurt, rollskew


# ---------------------------------------------------------------------------
# rollskew
# ---------------------------------------------------------------------------


def test_skew_zero_for_symmetric():
    """Perfectly symmetric window → skew == 0."""
    x = np.array([1.0, 2, 3, 4, 5])  # symmetric around 3
    result = rollskew(x, 5)
    assert abs(result[-1]) < 1e-12


def test_skew_positive_for_right_skewed():
    x = np.array([1.0, 1, 1, 1, 10])  # long right tail
    result = rollskew(x, 5)
    assert result[-1] > 0.5


def test_skew_negative_for_left_skewed():
    x = np.array([1.0, 10, 10, 10, 10])  # long left tail
    result = rollskew(x, 5)
    assert result[-1] < -0.5


def test_skew_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [5, 20, 50]:
        result = rollskew(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).skew().values
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_skew_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=15, replace=False)] = np.nan
    for w in [10, 30]:
        result = rollskew(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).skew().values
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_skew_matches_pandas_large_magnitude(rng):
    """Shift trick should keep precision on price-scale inputs."""
    x = rng.uniform(3900, 4100, size=300)
    result = rollskew(x, 20)
    reference = pd.Series(x).rolling(20, min_periods=20).skew().values
    np.testing.assert_allclose(result, reference, atol=1e-6, equal_nan=True)


def test_skew_window_lt_3_raises():
    from kuant.errors import KuantValueError

    x = np.arange(10, dtype=np.float64)
    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        rollskew(x, 2)


def test_skew_shift_invariance(rng):
    x = rng.uniform(-1, 1, size=100)
    r1 = rollskew(x, 20)
    r2 = rollskew(x + 100, 20)
    np.testing.assert_allclose(r1, r2, atol=1e-8, equal_nan=True)


def test_skew_scale_invariance_positive(rng):
    x = rng.uniform(-1, 1, size=100)
    r1 = rollskew(x, 20)
    r2 = rollskew(x * 5.0, 20)
    np.testing.assert_allclose(r1, r2, atol=1e-10, equal_nan=True)


def test_skew_negation_flips_sign(rng):
    x = rng.uniform(-1, 1, size=100)
    r1 = rollskew(x, 20)
    r2 = rollskew(-x, 20)
    np.testing.assert_allclose(r1, -r2, atol=1e-10, equal_nan=True)


def test_skew_constant_window_nan():
    x = np.full(10, 5.0)
    result = rollskew(x, 5)
    finite = ~np.isnan(result)
    # All windows constant → m2==0 → NaN
    assert not finite.any()


# ---------------------------------------------------------------------------
# rollkurt
# ---------------------------------------------------------------------------


def test_kurt_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [10, 30, 100]:
        result = rollkurt(x, w)
        reference = pd.Series(x).rolling(w, min_periods=w).kurt().values
        np.testing.assert_allclose(result, reference, atol=1e-9, equal_nan=True)


def test_kurt_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    x[rng.choice(300, size=15, replace=False)] = np.nan
    result = rollkurt(x, 30)
    reference = pd.Series(x).rolling(30, min_periods=30).kurt().values
    np.testing.assert_allclose(result, reference, atol=1e-9, equal_nan=True)


def test_kurt_matches_pandas_large_magnitude(rng):
    x = rng.uniform(3900, 4100, size=300)
    result = rollkurt(x, 30)
    reference = pd.Series(x).rolling(30, min_periods=30).kurt().values
    np.testing.assert_allclose(result, reference, atol=1e-6, equal_nan=True)


def test_kurt_window_lt_4_raises():
    from kuant.errors import KuantValueError

    x = np.arange(10, dtype=np.float64)
    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        rollkurt(x, 3)


def test_kurt_shift_invariance(rng):
    x = rng.uniform(-1, 1, size=100)
    r1 = rollkurt(x, 30)
    r2 = rollkurt(x + 100, 30)
    np.testing.assert_allclose(r1, r2, atol=1e-6, equal_nan=True)


def test_kurt_scale_invariance_positive(rng):
    x = rng.uniform(-1, 1, size=100)
    r1 = rollkurt(x, 30)
    r2 = rollkurt(x * 5.0, 30)
    np.testing.assert_allclose(r1, r2, atol=1e-9, equal_nan=True)


def test_kurt_constant_window_nan():
    x = np.full(10, 5.0)
    assert np.all(np.isnan(rollkurt(x, 5)))


# ---------------------------------------------------------------------------
# Shared edge / dtype / GPU
# ---------------------------------------------------------------------------


def test_window_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        rollskew(np.array([1.0, 2, 3, 4]), 0)
    with pytest.raises(ValueError, match="must be positive"):
        rollkurt(np.array([1.0, 2, 3, 4]), 0)


def test_2d_input_raises():
    with pytest.raises(ValueError, match="1D"):
        rollskew(np.array([[1.0, 2], [3, 4]]), 3)
    with pytest.raises(ValueError, match="1D"):
        rollkurt(np.array([[1.0, 2], [3, 4]]), 4)


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    assert rollskew(x, 3).dtype == np.float32
    assert rollkurt(x, 4).dtype == np.float32


def test_int_input_promoted():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert rollskew(x, 3).dtype == np.float64
    assert rollkurt(x, 4).dtype == np.float64


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    x = rng.uniform(-1, 1, size=200)
    np.testing.assert_allclose(
        rollskew(x, 20),
        cp.asnumpy(rollskew(cp.asarray(x), 20)),
        atol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rollkurt(x, 30),
        cp.asnumpy(rollkurt(cp.asarray(x), 30)),
        atol=1e-9,
        equal_nan=True,
    )

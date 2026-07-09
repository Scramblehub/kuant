"""Test suite for kuant.stats.zscore.

Since zscore composes rollmean and rollstd, tests focus on:
  1. Golden values — small hand-computed cases
  2. Reference match — pandas rolling z-score
  3. Zero-std policy — constant windows produce NaN
  4. Composition invariants — inherited from rollmean/rollstd
  5. CPU==GPU parity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.stats import zscore


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, w, ddof, expected",
    [
        # Arithmetic progression: trailing window means x[i] is always
        # one std above the window mean, so z = 1.0.
        ([1.0, 2, 3, 4, 5], 3, 1, [np.nan, np.nan, 1.0, 1.0, 1.0]),
        # Single spike: last value = 100 in [1,2,100], mean=34.33, std=~56.87
        # z = (100 - 34.33) / 56.87 = ~1.155
        (
            [1.0, 2, 100],
            3,
            1,
            [np.nan, np.nan, (100 - (1 + 2 + 100) / 3) / np.std([1, 2, 100], ddof=1)],
        ),
    ],
)
def test_golden_values(x, w, ddof, expected):
    result = zscore(np.array(x, dtype=np.float64), w, ddof=ddof)
    np.testing.assert_allclose(result, expected, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# 2. Reference match — pandas
# ---------------------------------------------------------------------------


def _pandas_zscore(x, w, ddof=1):
    """Reference: (x - rollmean) / rollstd via pandas."""
    s = pd.Series(x)
    rmean = s.rolling(w, min_periods=w).mean()
    rstd = s.rolling(w, min_periods=w).std(ddof=ddof)
    return ((s - rmean) / rstd).values


def test_matches_pandas_uniform(rng):
    x = rng.uniform(-1, 1, size=500)
    for w in [2, 5, 20, 50]:
        result = zscore(x, w)
        reference = _pandas_zscore(x, w)
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_matches_pandas_with_nans(rng):
    x = rng.uniform(-1, 1, size=300)
    nan_positions = rng.choice(300, size=15, replace=False)
    x[nan_positions] = np.nan
    for w in [3, 10, 30]:
        result = zscore(x, w)
        reference = _pandas_zscore(x, w)
        np.testing.assert_allclose(result, reference, atol=1e-10, equal_nan=True)


def test_matches_pandas_price_scale(rng):
    """Large-magnitude inputs. Tighter than rollstd's tolerance because
    the numerator and denominator both scale similarly."""
    x = rng.uniform(3900, 4100, size=300)
    for w in [5, 20]:
        result = zscore(x, w)
        reference = _pandas_zscore(x, w)
        np.testing.assert_allclose(result, reference, atol=1e-6, equal_nan=True)


# ---------------------------------------------------------------------------
# 3. Zero-std policy
# ---------------------------------------------------------------------------


def test_constant_window_is_nan():
    """Constant windows have std=0 → z=NaN by convention."""
    x = np.array([5.0, 5, 5, 5, 5, 6])
    result = zscore(x, 3)
    # Windows [5,5,5] all have std=0 → NaN
    # Window [5,5,6] has std>0 → finite
    assert np.isnan(result[2])
    assert np.isnan(result[3])
    assert np.isnan(result[4])
    # Last window includes a spike
    assert not np.isnan(result[5])


def test_all_constant_all_nan():
    x = np.full(10, 3.14)
    result = zscore(x, 4)
    assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# 4. Composition invariants
# ---------------------------------------------------------------------------


def test_first_w_minus_1_nan(rng):
    x = rng.uniform(-1, 1, size=100)
    for w in [2, 5, 20]:
        result = zscore(x, w)
        assert np.all(np.isnan(result[: w - 1]))


def test_result_length_equals_input(rng):
    x = rng.uniform(-1, 1, size=87)
    for w in [2, 10, 50]:
        assert zscore(x, w).size == 87


def test_shift_invariance(rng):
    """Adding a constant should not change zscore."""
    x = rng.uniform(-1, 1, size=100)
    r1 = zscore(x, 10)
    r2 = zscore(x + 1000, 10)
    np.testing.assert_allclose(r1, r2, atol=1e-8, equal_nan=True)


def test_scale_invariance(rng):
    """Scaling by a positive constant should not change zscore."""
    x = rng.uniform(0.1, 1.0, size=100)  # avoid crossing 0
    r1 = zscore(x, 10)
    r2 = zscore(x * 7.0, 10)
    np.testing.assert_allclose(r1, r2, atol=1e-10, equal_nan=True)


def test_int_input_promoted_to_float64():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = zscore(x, 3)
    assert result.dtype == np.float64


def test_dtype_preserved_float32():
    x = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
    result = zscore(x, 3)
    assert result.dtype == np.float32


def test_2d_input_raises():
    with pytest.raises(ValueError, match="1D"):
        zscore(np.array([[1.0, 2], [3, 4]]), 2)


def test_window_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        zscore(np.array([1.0, 2, 3]), 0)


def test_python_list_input():
    result = zscore([1.0, 2, 3, 4, 5], 3)
    np.testing.assert_allclose(result[2:], [1, 1, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# 5. CPU == GPU
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    x_cpu = rng.uniform(-1, 1, size=500)
    x_gpu = cp.asarray(x_cpu)
    for w in [3, 20, 50]:
        r_cpu = zscore(x_cpu, w)
        r_gpu = cp.asnumpy(zscore(x_gpu, w))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, equal_nan=True)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp

    result = zscore(cp.asarray([1.0, 2, 3, 4, 5]), 3)
    assert isinstance(result, cp.ndarray)


def test_gpu_constant_window(skip_no_gpu):
    import cupy as cp

    x = cp.array([5.0, 5, 5, 5, 5])
    result = zscore(x, 3)
    # All windows constant → all NaN
    assert cp.all(cp.isnan(result))

"""Test suite for kuant.core.normcdf.

Structure matches the 5-validation strategy from the plan:
  1. Golden values  — hardcoded x → Φ(x) pairs to eyeball
  2. Reference match — matches scipy.special.ndtr to eps
  3. Edge cases     — NaN, inf, empty, scalar, int, 2D
  4. Property tests — symmetry, monotonicity, range
  5. CPU==GPU       — GPU path bit-close to CPU path

Additionally: an auto-fixture (in conftest.py) checks that no test leaks
GPU memory above the 100MB threshold. Any test here that grows the memory
pool by more than that fails automatically.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from kuant.core import normcdf


# ---------------------------------------------------------------------------
# 1. Golden values — hand-verified against textbooks and online tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145707),
        (2.0, 0.9772498680518208),
        (-2.0, 0.022750131948179195),
        (1.96, 0.9750021048517795),  # famous 97.5th percentile
        (-1.96, 0.024997895148220435),
        (3.0, 0.9986501019683699),
        (5.0, 0.9999997133484281),
    ],
)
def test_golden_values(x, expected):
    """Hardcoded reference values for eyeball verification."""
    result = normcdf(x)
    assert isinstance(result, float), "scalar input → scalar output"
    assert result == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# 2. Reference match — batched against scipy.special.ndtr
# ---------------------------------------------------------------------------


def test_matches_scipy_ndtr_uniform(rng):
    """Uniform sample across [-4, 4] matches scipy to double precision."""
    x = rng.uniform(-4, 4, size=10_000)
    result = normcdf(x)
    reference = ndtr(x)
    np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)


def test_matches_scipy_ndtr_extreme(rng):
    """Wide-tail sample [-20, 20] — tests saturation behavior."""
    x = rng.uniform(-20, 20, size=5_000)
    result = normcdf(x)
    reference = ndtr(x)
    # Looser tolerance near saturation (both should be effectively 0 or 1)
    np.testing.assert_allclose(result, reference, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_nan_passthrough():
    """NaN in → NaN out. Never crashes."""
    result = normcdf(float("nan"))
    assert np.isnan(result)


def test_positive_infinity():
    """+∞ → 1.0."""
    assert normcdf(float("inf")) == 1.0


def test_negative_infinity():
    """-∞ → 0.0."""
    assert normcdf(float("-inf")) == 0.0


def test_empty_array():
    """Empty array input → empty array output (no crash)."""
    x = np.array([], dtype=np.float64)
    result = normcdf(x)
    assert isinstance(result, np.ndarray)
    assert result.size == 0
    assert result.dtype == np.float64


def test_scalar_int_input():
    """Int scalar input → float output."""
    result = normcdf(0)  # Python int
    assert isinstance(result, float)
    assert result == pytest.approx(0.5, abs=1e-12)


def test_array_int_input():
    """Int array input → cast to float64."""
    x = np.array([-1, 0, 1], dtype=np.int64)
    result = normcdf(x)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [0.15865525393145707, 0.5, 0.8413447460685429], atol=1e-12)


def test_2d_array_preserves_shape():
    """2D input → 2D output with same shape."""
    x = np.array([[0.0, 1.0], [-1.0, 2.0]])
    result = normcdf(x)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(
        result,
        [[0.5, 0.8413447460685429], [0.15865525393145707, 0.9772498680518208]],
        atol=1e-12,
    )


def test_3d_array_preserves_shape(rng):
    """3D input → 3D output. Reshape/ravel path validated."""
    x = rng.uniform(-3, 3, size=(4, 5, 6))
    result = normcdf(x)
    assert result.shape == (4, 5, 6)
    np.testing.assert_allclose(result, ndtr(x), atol=1e-12)


def test_dtype_preserved_float32(rng):
    """float32 in → float32 out (dtype not silently promoted)."""
    x = rng.uniform(-3, 3, size=100).astype(np.float32)
    result = normcdf(x)
    assert result.dtype == np.float32
    # Looser tolerance for float32
    np.testing.assert_allclose(result, ndtr(x), atol=1e-6, rtol=1e-6)


def test_dtype_preserved_float64(rng):
    """float64 in → float64 out."""
    x = rng.uniform(-3, 3, size=100).astype(np.float64)
    result = normcdf(x)
    assert result.dtype == np.float64


def test_list_input():
    """Python list input works (auto-array coercion)."""
    result = normcdf([-1.0, 0.0, 1.0])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.15865525393145707, 0.5, 0.8413447460685429], atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_symmetry(rng):
    """Φ(-x) == 1 - Φ(x) for all finite x."""
    x = rng.uniform(-6, 6, size=1_000)
    lhs = normcdf(-x)
    rhs = 1.0 - normcdf(x)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_monotonically_non_decreasing(rng):
    """Φ is non-decreasing: sort input, output should also be non-decreasing."""
    x = np.sort(rng.uniform(-5, 5, size=1_000))
    result = normcdf(x)
    diffs = np.diff(result)
    # All differences should be ≥ 0 (allow tiny numerical noise)
    assert np.all(diffs >= -1e-15)


def test_output_in_unit_interval(rng):
    """Φ ∈ [0, 1] for every input."""
    x = rng.uniform(-20, 20, size=1_000)
    result = normcdf(x)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_at_zero_is_half():
    """Φ(0) = 0.5 exactly."""
    assert normcdf(0.0) == pytest.approx(0.5, abs=1e-15)


# ---------------------------------------------------------------------------
# 5. CPU == GPU path (only runs if a GPU is available)
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    """cupy input should produce same result as numpy input."""
    import cupy as cp

    x_cpu = rng.uniform(-4, 4, size=10_000)
    x_gpu = cp.asarray(x_cpu)

    result_cpu = normcdf(x_cpu)
    result_gpu = normcdf(x_gpu)

    # Bring GPU result back to CPU for comparison
    result_gpu_as_np = cp.asnumpy(result_gpu)

    # cupy erf and scipy ndtr may differ by a few ULP; 1e-12 is tight but
    # achievable for library implementations that follow the same math.
    np.testing.assert_allclose(result_cpu, result_gpu_as_np, atol=1e-12)


def test_gpu_preserves_backend(skip_no_gpu, rng):
    """cupy in → cupy out (not silently transferred to CPU)."""
    import cupy as cp

    x = cp.asarray(rng.uniform(-2, 2, size=100))
    result = normcdf(x)
    assert isinstance(result, cp.ndarray), "GPU input should stay on GPU"

"""Test suite for kuant.core.normppf.

Peter Acklam's rational approximation, accurate to ~1.15e-9.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from kuant.core import normcdf, normppf


# ---------------------------------------------------------------------------
# 1. Match scipy.stats.norm.ppf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p", [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999]
)
def test_matches_scipy(p):
    assert abs(normppf(p) - norm.ppf(p)) < 5e-9


# ---------------------------------------------------------------------------
# 2. Round-trip: normppf(normcdf(x)) == x
# ---------------------------------------------------------------------------


def test_round_trip_via_normcdf(rng):
    xs = rng.uniform(-6, 6, 200)
    ps = normcdf(xs)
    recovered = normppf(ps)
    # Round-trip error dominated by Acklam ~1.15e-9
    np.testing.assert_allclose(recovered, xs, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Boundary values
# ---------------------------------------------------------------------------


def test_p_half_is_zero():
    assert normppf(0.5) == 0.0


def test_symmetry():
    """normppf(1 - p) = -normppf(p)."""
    for p in [0.01, 0.1, 0.25, 0.4]:
        assert abs(normppf(1 - p) + normppf(p)) < 5e-9


def test_p_zero_is_minus_inf():
    assert normppf(0.0) == -np.inf


def test_p_one_is_plus_inf():
    assert normppf(1.0) == np.inf


def test_p_out_of_range_returns_nan():
    assert np.isnan(normppf(-0.1))
    assert np.isnan(normppf(1.5))


def test_p_nan_returns_nan():
    assert np.isnan(normppf(np.nan))


# ---------------------------------------------------------------------------
# 4. Region coverage: central, lower tail, upper tail
# ---------------------------------------------------------------------------


def test_lower_tail_region():
    """p < 0.02425 uses the lower-tail branch."""
    for p in [0.001, 0.005, 0.02]:
        assert abs(normppf(p) - norm.ppf(p)) < 5e-9


def test_upper_tail_region():
    """p > 0.97575 uses the upper-tail branch."""
    for p in [0.98, 0.995, 0.999]:
        assert abs(normppf(p) - norm.ppf(p)) < 5e-9


def test_central_region():
    for p in [0.05, 0.5, 0.95]:
        assert abs(normppf(p) - norm.ppf(p)) < 5e-9


# ---------------------------------------------------------------------------
# 5. Batched
# ---------------------------------------------------------------------------


def test_batched_matches_scipy(rng):
    ps = rng.uniform(1e-5, 1 - 1e-5, 500)
    ours = normppf(ps)
    ref = norm.ppf(ps)
    np.testing.assert_allclose(ours, ref, atol=5e-9)


def test_batched_int_promoted():
    """Integer p (only 0 and 1 are valid, but check they promote cleanly)."""
    ps = np.array([0, 1], dtype=np.int64)
    result = normppf(ps)
    assert result.dtype == np.float64
    assert result[0] == -np.inf
    assert result[1] == np.inf


# ---------------------------------------------------------------------------
# 6. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    ps = rng.uniform(1e-5, 1 - 1e-5, 100)
    r_cpu = normppf(ps)
    r_gpu = cp.asnumpy(normppf(cp.asarray(ps)))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-13)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp

    result = normppf(cp.asarray([0.1, 0.5, 0.9]))
    assert isinstance(result, cp.ndarray)

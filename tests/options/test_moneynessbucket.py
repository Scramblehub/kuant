"""Test suite for kuant.options.moneynessbucket."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.options import moneynessbucket


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


def test_default_edges_partition():
    """Default edges [-0.10, -0.03, 0.03, 0.10] → 5 buckets."""
    S = np.array([100.0] * 5)
    # With r=0.05, T=1: F ≈ 105.13. K = 80 → m = -0.273 (deep ITM call)
    # K = 97  → m ≈ -0.081 (ITM)
    # K = 100 → m ≈ -0.050 (still ITM by fwd-log)
    # K = 103 → m ≈ -0.020 (ATM)
    # K = 120 → m ≈  0.132 (deep OTM)
    K = np.array([80.0, 97.0, 100.0, 103.0, 120.0])
    result = moneynessbucket(S, K, 1.0, 0.05)
    np.testing.assert_array_equal(result, [0, 1, 1, 2, 4])


def test_ATM_forward_maps_to_center():
    """K = F → m = 0 → bucket 2 (ATM with default edges)."""
    S, r, T = 100.0, 0.05, 1.0
    F = S * np.exp(r * T)
    K = np.array([F])
    result = moneynessbucket(np.array([S]), K, T, r)
    assert result[0] == 2


def test_custom_edges():
    S = np.array([100.0] * 3)
    K = np.array([90.0, 100.0, 110.0])
    result = moneynessbucket(S, K, 1.0, 0.05, edges=np.array([-0.05, 0.05]))
    # F ≈ 105.13, m = [-0.153, -0.050, +0.045] → [0, 0, 1]
    np.testing.assert_array_equal(result, [0, 0, 1])


def test_with_dividend():
    """Dividend q reduces forward: F = S·e^((r-q)T). Test consistency."""
    S = np.array([100.0])
    K = np.array([100.0])
    T, r, q = 1.0, 0.05, 0.02
    # F = S·exp((r-q)T) ≈ 103.04, so K/F ≈ 0.9705, m ≈ -0.030
    result = moneynessbucket(S, K, T, r, q)
    assert result[0] in (1, 2)  # right around bucket boundary


# ---------------------------------------------------------------------------
# 2. Boundary behavior
# ---------------------------------------------------------------------------


def test_exactly_on_edge():
    """np.digitize: `bins[i-1] <= x < bins[i]` → i. Exact-edge → right bin.

    Floating-point construction of m ≈ -0.10 may land on either side of
    the edge; accept both 0 and 1 as valid.
    """
    S = np.array([100.0])
    T, r = 1.0, 0.05
    F = S[0] * np.exp(r * T)
    K = np.array([F * np.exp(-0.10)])  # m = -0.10 target
    result = moneynessbucket(S, K, T, r)
    assert result[0] in (0, 1)


# ---------------------------------------------------------------------------
# 3. Broadcasting
# ---------------------------------------------------------------------------


def test_broadcasting_scalar_S():
    K = np.array([80.0, 100.0, 120.0])
    result = moneynessbucket(100.0, K, 1.0, 0.05)
    assert result.shape == (3,)


def test_broadcasting_2d():
    S = np.array([[100.0, 100.0, 100.0]])
    K = np.array([[80.0], [100.0], [120.0]])
    result = moneynessbucket(S, K, 1.0, 0.05)
    assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# 4. Errors
# ---------------------------------------------------------------------------


def test_2d_edges_raises():
    S = np.array([100.0])
    K = np.array([100.0])
    with pytest.raises(ValueError, match="1D"):
        moneynessbucket(S, K, 1.0, 0.05, edges=np.array([[-0.1, 0.1]]))


# ---------------------------------------------------------------------------
# 5. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    S = rng.uniform(50, 150, 50)
    K = rng.uniform(50, 150, 50)
    T = rng.uniform(0.1, 2.0, 50)
    r_cpu = moneynessbucket(S, K, T, 0.05)
    r_gpu = cp.asnumpy(moneynessbucket(cp.asarray(S), cp.asarray(K), cp.asarray(T), 0.05))
    np.testing.assert_array_equal(r_cpu, r_gpu)

"""Test suite for kuant.options.putpayoff."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.options import callpayoff, putpayoff


@pytest.mark.parametrize(
    "S, K, expected",
    [
        (80.0, 100.0, 20.0),
        (100.0, 100.0, 0.0),
        (120.0, 100.0, 0.0),
        (0.0, 100.0, 100.0),  # S=0 -> full payoff
        (100.0, 0.0, 0.0),  # K=0 -> worthless
    ],
)
def test_golden(S, K, expected):
    assert putpayoff(S, K) == expected


def test_S_zero_gives_K():
    for K in [50.0, 100.0, 250.0]:
        assert putpayoff(0.0, K) == K


def test_at_the_money_zero():
    assert putpayoff(100.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# Parity check: intrinsic call - intrinsic put = S - K
# ---------------------------------------------------------------------------


def test_intrinsic_parity(rng):
    """Basic identity: callpayoff(S,K) - putpayoff(S,K) = S - K."""
    S = rng.uniform(0, 200, 100)
    K = rng.uniform(0, 200, 100)
    lhs = callpayoff(S, K) - putpayoff(S, K)
    rhs = S - K
    np.testing.assert_allclose(lhs, rhs, atol=1e-14)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


def test_S_array_K_scalar():
    S = np.array([80.0, 100.0, 120.0, 50.0])
    result = putpayoff(S, 100.0)
    np.testing.assert_array_equal(result, [20.0, 0.0, 0.0, 50.0])


def test_broadcast_2d():
    S = np.array([[80.0, 100.0, 120.0]])
    K = np.array([[110.0], [90.0]])
    result = putpayoff(S, K)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, [[30.0, 10.0, 0.0], [10.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------


def test_float32_preserved():
    S = np.array([100.0, 80.0], dtype=np.float32)
    K = np.array([100.0, 100.0], dtype=np.float32)
    assert putpayoff(S, K).dtype == np.float32


def test_int_promoted_to_float64():
    S = np.array([80, 100], dtype=np.int64)
    K = np.array([100, 100], dtype=np.int64)
    result = putpayoff(S, K)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------


def test_non_negative(rng):
    S = rng.uniform(0, 200, 100)
    K = rng.uniform(0, 200, 100)
    assert np.all(putpayoff(S, K) >= 0)


def test_matches_naive_max():
    S = np.array([80.0, 100.0, 120.0])
    K = 100.0
    np.testing.assert_array_equal(putpayoff(S, K), np.maximum(K - S, 0.0))


# ---------------------------------------------------------------------------
# GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp

    S = rng.uniform(0, 200, 100)
    K = rng.uniform(0, 200, 100)
    r_cpu = putpayoff(S, K)
    r_gpu = cp.asnumpy(putpayoff(cp.asarray(S), cp.asarray(K)))
    np.testing.assert_array_equal(r_cpu, r_gpu)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp

    result = putpayoff(cp.asarray([80.0, 100.0]), cp.asarray([100.0, 100.0]))
    assert isinstance(result, cp.ndarray)

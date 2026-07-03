'''Test suite for kuant.options.callpayoff.'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.options import callpayoff


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'S, K, expected',
    [
        (120.0, 100.0, 20.0),
        (100.0, 100.0, 0.0),
        (80.0, 100.0, 0.0),
        (0.0, 100.0, 0.0),      # far OTM
        (200.0, 0.0, 200.0),    # zero strike
    ],
)
def test_golden(S, K, expected):
    assert callpayoff(S, K) == expected


def test_zero_strike_equals_spot():
    for S in [50.0, 100.0, 250.0]:
        assert callpayoff(S, 0.0) == S


def test_at_the_money_zero():
    assert callpayoff(100.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# 2. Broadcasting
# ---------------------------------------------------------------------------


def test_S_array_K_scalar():
    S = np.array([80.0, 100.0, 120.0, 150.0])
    result = callpayoff(S, 100.0)
    np.testing.assert_array_equal(result, [0.0, 0.0, 20.0, 50.0])


def test_K_array_S_scalar():
    K = np.array([80.0, 100.0, 120.0])
    result = callpayoff(100.0, K)
    np.testing.assert_array_equal(result, [20.0, 0.0, 0.0])


def test_both_arrays_same_shape():
    S = np.array([80.0, 100.0, 120.0])
    K = np.array([90.0, 110.0, 100.0])
    result = callpayoff(S, K)
    np.testing.assert_array_equal(result, [0.0, 0.0, 20.0])


def test_broadcast_2d():
    S = np.array([[80.0, 100.0, 120.0]])       # (1, 3)
    K = np.array([[90.0], [110.0]])            # (2, 1)
    result = callpayoff(S, K)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, [[0.0, 10.0, 30.0],
                                            [0.0,  0.0, 10.0]])


# ---------------------------------------------------------------------------
# 3. dtype preservation
# ---------------------------------------------------------------------------


def test_float32_preserved():
    S = np.array([100.0, 120.0], dtype=np.float32)
    K = np.array([100.0, 100.0], dtype=np.float32)
    assert callpayoff(S, K).dtype == np.float32


def test_int_promoted_to_float64():
    S = np.array([100, 120], dtype=np.int64)
    K = np.array([100, 100], dtype=np.int64)
    result = callpayoff(S, K)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [0.0, 20.0])


def test_python_scalars():
    assert callpayoff(120, 100) == 20.0
    assert isinstance(callpayoff(120, 100), float)


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_non_negative(rng):
    '''Payoff never < 0.'''
    S = rng.uniform(0, 200, 100)
    K = rng.uniform(0, 200, 100)
    assert np.all(callpayoff(S, K) >= 0)


def test_matches_naive_max():
    '''Semantic sanity — same as np.maximum.'''
    S = np.array([80.0, 100.0, 120.0, 150.0])
    K = 100.0
    np.testing.assert_array_equal(callpayoff(S, K), np.maximum(S - K, 0.0))


# ---------------------------------------------------------------------------
# 5. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    S = rng.uniform(0, 200, 100)
    K = rng.uniform(0, 200, 100)
    r_cpu = callpayoff(S, K)
    r_gpu = cp.asnumpy(callpayoff(cp.asarray(S), cp.asarray(K)))
    np.testing.assert_array_equal(r_cpu, r_gpu)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    result = callpayoff(cp.asarray([100.0, 120.0]), cp.asarray([100.0, 100.0]))
    assert isinstance(result, cp.ndarray)

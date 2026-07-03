'''Test suite for kuant.core.logsumexp.'''
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_lse

from kuant.core import logsumexp


# ---------------------------------------------------------------------------
# 1. Match scipy.special.logsumexp
# ---------------------------------------------------------------------------


def test_matches_scipy_1d(rng):
    for _ in range(10):
        x = rng.uniform(-100, 100, 50)
        assert abs(logsumexp(x) - scipy_lse(x)) < 1e-10


def test_matches_scipy_extreme_values():
    # Large positive: naive exp overflows
    x = np.array([1000.0, 1000.0])
    assert abs(logsumexp(x) - (1000.0 + np.log(2))) < 1e-10
    # Large negative: still stable
    x = np.array([-1000.0, -1000.0])
    assert abs(logsumexp(x) - (-1000.0 + np.log(2))) < 1e-10


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------


def test_all_minus_inf_returns_minus_inf():
    result = logsumexp(np.array([-np.inf, -np.inf, -np.inf]))
    assert result == -np.inf


def test_single_element():
    assert logsumexp(np.array([3.5])) == 3.5


def test_with_finite_and_neg_inf_mix():
    '''logsumexp([0, -inf, -inf]) = log(exp(0)) = 0.'''
    result = logsumexp(np.array([0.0, -np.inf, -np.inf]))
    assert abs(result) < 1e-14


def test_int_promoted_to_float64():
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = logsumexp(x)
    ref = scipy_lse(x)
    assert abs(result - ref) < 1e-10


# ---------------------------------------------------------------------------
# 3. Axis reduction
# ---------------------------------------------------------------------------


def test_axis_0_reduction(rng):
    x = rng.uniform(-10, 10, (5, 8))
    ours = logsumexp(x, axis=0)
    ref = scipy_lse(x, axis=0)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_axis_1_reduction(rng):
    x = rng.uniform(-10, 10, (5, 8))
    ours = logsumexp(x, axis=1)
    ref = scipy_lse(x, axis=1)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_keepdims(rng):
    x = rng.uniform(-10, 10, (5, 8))
    ours = logsumexp(x, axis=1, keepdims=True)
    ref = scipy_lse(x, axis=1, keepdims=True)
    assert ours.shape == (5, 1)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Numerical stability
# ---------------------------------------------------------------------------


def test_no_overflow_at_1e300():
    '''Values around log(max float) — naive exp would overflow.'''
    x = np.array([700.0, 700.0])  # exp(700) is near float64 max
    result = logsumexp(x)
    ref = 700.0 + np.log(2)
    assert abs(result - ref) < 1e-10


def test_associativity_via_partition(rng):
    '''logsumexp([logsumexp(A), logsumexp(B)]) == logsumexp(A ∪ B).'''
    x = rng.uniform(-50, 50, 100)
    full = logsumexp(x)
    partial = logsumexp(np.array([logsumexp(x[:50]), logsumexp(x[50:])]))
    assert abs(full - partial) < 1e-10


# ---------------------------------------------------------------------------
# 5. No warnings raised for -inf case (via errstate suppression)
# ---------------------------------------------------------------------------


def test_no_warnings_all_neg_inf():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # raise on any warning
        result = logsumexp(np.array([-np.inf, -np.inf]))
        assert result == -np.inf


# ---------------------------------------------------------------------------
# 6. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    x = rng.uniform(-100, 100, 500)
    r_cpu = logsumexp(x)
    r_gpu = float(logsumexp(cp.asarray(x)))
    assert abs(r_cpu - r_gpu) < 1e-10


def test_gpu_axis_reduction(skip_no_gpu, rng):
    import cupy as cp
    x = rng.uniform(-10, 10, (5, 8))
    r_cpu = logsumexp(x, axis=1)
    r_gpu = cp.asnumpy(logsumexp(cp.asarray(x), axis=1))
    np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10)

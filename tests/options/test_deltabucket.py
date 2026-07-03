'''Test suite for kuant.options.deltabucket.'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.options import deltabucket


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


def test_scalar_target_returns_int():
    deltas = np.array([0.05, 0.15, 0.25, 0.50, 0.75, 0.95])
    result = deltabucket(deltas, 0.25)
    assert result == 2
    assert isinstance(result, int)


def test_array_target_returns_array():
    deltas = np.array([0.05, 0.15, 0.25, 0.50, 0.75, 0.95])
    result = deltabucket(deltas, np.array([0.10, 0.50, 0.90]))
    np.testing.assert_array_equal(result, [1, 3, 5])


def test_target_between_two_options_picks_closer():
    deltas = np.array([0.20, 0.30])
    assert deltabucket(deltas, 0.24) == 0
    assert deltabucket(deltas, 0.26) == 1


def test_exact_match():
    deltas = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    assert deltabucket(deltas, 0.25) == 1


def test_target_outside_range_picks_boundary():
    deltas = np.array([0.10, 0.25, 0.50])
    assert deltabucket(deltas, 0.01) == 0
    assert deltabucket(deltas, 0.99) == 2


# ---------------------------------------------------------------------------
# 2. Signed deltas — put convention
# ---------------------------------------------------------------------------


def test_put_deltas_negative():
    '''Put deltas are negative; passing negative target works.'''
    put_deltas = np.array([-0.95, -0.75, -0.50, -0.25, -0.10, -0.05])
    # "25-delta put" means delta = -0.25
    assert deltabucket(put_deltas, -0.25) == 3
    assert deltabucket(put_deltas, -0.10) == 4


# ---------------------------------------------------------------------------
# 3. Errors
# ---------------------------------------------------------------------------


def test_2d_deltas_raises():
    with pytest.raises(ValueError, match='1D'):
        deltabucket(np.zeros((3, 3)), 0.25)


# ---------------------------------------------------------------------------
# 4. Multi-target performance sanity
# ---------------------------------------------------------------------------


def test_many_targets_at_once(rng):
    deltas = np.sort(rng.uniform(0, 1, 100))
    targets = rng.uniform(0, 1, 50)
    result = deltabucket(deltas, targets)
    assert result.shape == (50,)
    # Verify each result really is the argmin
    for i, t in enumerate(targets):
        expected = int(np.argmin(np.abs(deltas - t)))
        assert result[i] == expected


# ---------------------------------------------------------------------------
# 5. GPU parity
# ---------------------------------------------------------------------------


def test_gpu_matches_cpu(skip_no_gpu, rng):
    import cupy as cp
    deltas = np.sort(rng.uniform(0, 1, 50))
    targets = rng.uniform(0, 1, 10)
    r_cpu = deltabucket(deltas, targets)
    r_gpu = cp.asnumpy(deltabucket(cp.asarray(deltas), cp.asarray(targets)))
    np.testing.assert_array_equal(r_cpu, r_gpu)


def test_gpu_preserves_backend(skip_no_gpu):
    import cupy as cp
    deltas = cp.asarray([0.10, 0.25, 0.50])
    result = deltabucket(deltas, cp.asarray([0.20]))
    assert isinstance(result, cp.ndarray)

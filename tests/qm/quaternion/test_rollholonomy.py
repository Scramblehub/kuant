"""Tests for kuant.qm.quaternion.rollholonomy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kuant.errors import KuantNumericWarning, KuantShapeError, KuantValueError
from kuant.qm.quaternion import Quaternion, rollholonomy


def _rot(axis, angle) -> np.ndarray:
    return Quaternion.from_axis_angle(axis, angle).as_array()


def test_warmup_first_w_minus_1_are_nan():
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (10, 1))
    hol, ang = rollholonomy(panel, window=3)
    assert np.isnan(hol[:2]).all()
    assert np.isnan(ang[:2]).all()
    assert np.isfinite(hol[2:]).all()


def test_uniform_stream_composes_expected_angle():
    """A trailing window of 3 30-degree rotations composes to 90 degrees."""
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (10, 1))
    _, ang = rollholonomy(panel, window=3)
    for t in range(2, 10):
        assert abs(ang[t] - math.pi / 2) < 1e-10


def test_identity_stream_returns_identity_and_zero_angle():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    panel = np.tile(identity, (10, 1))
    hol, ang = rollholonomy(panel, window=5)
    for t in range(4, 10):
        np.testing.assert_allclose(hol[t], identity, atol=1e-12)
        assert ang[t] < 1e-10


def test_full_2pi_window_returns_to_identity():
    """A window that spans 360 degrees of rotation has angle 0 holonomy."""
    q = _rot([0, 0, 1], math.pi / 4)
    panel = np.tile(q, (12, 1))
    _, ang = rollholonomy(panel, window=8)
    for t in range(7, 12):
        assert ang[t] < 1e-10


def test_axis_switching_produces_curvature():
    """Rotations around alternating axes accumulate nontrivial holonomy."""
    q_x = _rot([1, 0, 0], math.pi / 6)
    q_y = _rot([0, 1, 0], math.pi / 6)
    panel = np.stack([q_x, q_y, q_x, q_y, q_x, q_y, q_x, q_y])
    _, ang = rollholonomy(panel, window=4)
    # Alternating small rotations around different axes: the composed
    # rotation is not zero.
    for t in range(3, 8):
        assert ang[t] > 0.1


def test_nan_row_propagates_within_window():
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (10, 1))
    panel[5] = np.nan
    _, ang = rollholonomy(panel, window=3)
    # Rows 5, 6, 7 all have the NaN in their trailing 3-bar window.
    assert np.isnan(ang[5])
    assert np.isnan(ang[6])
    assert np.isnan(ang[7])
    # Row 8: window is bars 6..8, no NaN.
    assert np.isfinite(ang[8])


def test_window_exceeds_data_returns_all_nan_and_warns():
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (3, 1))
    with pytest.warns(KuantNumericWarning, match="KW-VAL-WINDOW-EXCEEDS-DATA"):
        hol, ang = rollholonomy(panel, window=10)
    assert np.isnan(hol).all()
    assert np.isnan(ang).all()


def test_reject_1d_input():
    with pytest.raises(KuantShapeError):
        rollholonomy(np.array([1.0, 0.0, 0.0, 0.0]), window=3)


def test_reject_bad_last_axis_length():
    with pytest.raises(KuantShapeError):
        rollholonomy(np.zeros((10, 3)), window=3)


def test_reject_nonpositive_window():
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (10, 1))
    with pytest.raises(KuantValueError):
        rollholonomy(panel, window=0)


def test_output_shape():
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (10, 1))
    hol, ang = rollholonomy(panel, window=3)
    assert hol.shape == (10, 4)
    assert ang.shape == (10,)


def test_angle_is_within_0_pi():
    """Rolling angle magnitude is bounded in [0, pi]."""
    q = _rot([1, 1, 1], 0.7)
    panel = np.tile(q, (20, 1))
    _, ang = rollholonomy(panel, window=5)
    finite = ang[np.isfinite(ang)]
    assert (finite >= -1e-12).all()
    assert (finite <= math.pi + 1e-12).all()

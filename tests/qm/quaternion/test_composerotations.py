"""Tests for kuant.qm.quaternion.composerotations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.qm.quaternion import Quaternion, composerotations, quat_angle, quaternion_distance


def _rot(axis, angle) -> np.ndarray:
    return Quaternion.from_axis_angle(axis, angle).as_array()


def test_single_element_returns_input():
    q = _rot([0, 0, 1], math.pi / 4)
    Q = composerotations(q[None, :])
    np.testing.assert_allclose(Q, q, atol=1e-12)


def test_two_z_rotations_add_angles():
    """Two 45-degree z-rotations compose to a 90-degree z-rotation."""
    q1 = _rot([0, 0, 1], math.pi / 4)
    q2 = _rot([0, 0, 1], math.pi / 4)
    Q = composerotations(np.stack([q1, q2]))
    expected = _rot([0, 0, 1], math.pi / 2)
    assert quaternion_distance(Q, expected) < 1e-10


def test_full_2pi_returns_to_identity():
    """Eight 45-degree z-rotations = 360 degrees = identity (up to sign)."""
    q = _rot([0, 0, 1], math.pi / 4)
    panel = np.tile(q, (8, 1))
    Q = composerotations(panel)
    # Angle of the composition should be close to 0 (identity).
    assert float(quat_angle(Q)) < 1e-10


def test_left_to_right_convention():
    """Compose(q_a, q_b) applies q_a first then q_b."""
    q_x = _rot([1, 0, 0], math.pi / 2)  # 90 around x
    q_z = _rot([0, 0, 1], math.pi / 2)  # 90 around z
    Q = composerotations(np.stack([q_x, q_z]))
    # Apply to y-hat: q_x sends y -> z, then q_z sends z -> z (unchanged).
    q_obj = Quaternion.from_array(Q)
    rotated = q_obj.rotate([0.0, 1.0, 0.0])
    np.testing.assert_allclose(rotated, [0, 0, 1], atol=1e-10)


def test_return_trajectory_shape():
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (5, 1))
    Q_final, traj = composerotations(panel, return_trajectory=True)
    assert traj.shape == (5, 4)
    np.testing.assert_allclose(traj[-1], Q_final, atol=1e-12)


def test_trajectory_accumulates():
    q = _rot([0, 0, 1], math.pi / 6)
    panel = np.tile(q, (3, 1))
    _, traj = composerotations(panel, return_trajectory=True)
    # Row t should be a rotation by (t + 1) * 30 degrees.
    for t in range(3):
        assert abs(float(quat_angle(traj[t])) - (t + 1) * math.pi / 6) < 1e-10


def test_accepts_list_of_quaternion_objects():
    quats = [Quaternion.from_axis_angle([0, 0, 1], math.pi / 6) for _ in range(3)]
    Q = composerotations(quats)
    assert abs(float(quat_angle(Q)) - math.pi / 2) < 1e-10


def test_reject_empty_input():
    with pytest.raises(KuantValueError, match="KE-VAL-EMPTY"):
        composerotations(np.empty((0, 4)))


def test_reject_bad_shape():
    with pytest.raises(KuantShapeError):
        composerotations(np.array([1.0, 0.0, 0.0, 0.0]))  # (4,) not (T, 4)

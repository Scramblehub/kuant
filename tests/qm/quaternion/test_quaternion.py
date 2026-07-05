"""Tests for kuant.qm.quaternion primitives + array ops + slerp + distance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.qm.quaternion import (
    Quaternion,
    quat_angle,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quaternion_distance,
    slerp,
)


# ---------- Quaternion dataclass --------------------------------------


def test_identity():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    assert q.w == 1.0
    assert q.angle() == 0.0


def test_auto_normalizes_non_unit_input():
    q = Quaternion(2.0, 0.0, 0.0, 0.0)
    # Should have been rescaled to (1, 0, 0, 0).
    assert abs(q.w - 1.0) < 1e-12


def test_rejects_zero_norm():
    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        Quaternion(0.0, 0.0, 0.0, 0.0)


def test_as_array_roundtrip():
    q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 3)
    arr = q.as_array()
    q2 = Quaternion.from_array(arr)
    assert abs(q2.w - q.w) < 1e-12
    assert abs(q2.z - q.z) < 1e-12


def test_from_array_rejects_bad_size():
    with pytest.raises(KuantShapeError, match="KE-SHAPE-EXPECTED"):
        Quaternion.from_array([1.0, 2.0, 3.0])


# ---------- algebra ---------------------------------------------------


def test_multiply_identity_is_noop():
    q = Quaternion.from_axis_angle([1, 0, 0], math.pi / 4)
    identity = Quaternion(1.0, 0.0, 0.0, 0.0)
    r = q.multiply(identity)
    assert abs(r.w - q.w) < 1e-12
    assert abs(r.x - q.x) < 1e-12


def test_multiply_non_commutative():
    """q * p != p * q in general."""
    q = Quaternion.from_axis_angle([1, 0, 0], math.pi / 3)
    p = Quaternion.from_axis_angle([0, 1, 0], math.pi / 3)
    left = q.multiply(p)
    right = p.multiply(q)
    # Angle-difference is a good scalar proxy.
    dist = quaternion_distance(left.as_array(), right.as_array())
    assert dist > 0.1


def test_conjugate_is_inverse_for_unit():
    q = Quaternion.from_axis_angle([1, 2, 3], math.pi / 5)
    r = q.multiply(q.conjugate())
    # Should be identity.
    assert abs(r.w - 1.0) < 1e-12
    assert abs(r.x) < 1e-12
    assert abs(r.y) < 1e-12
    assert abs(r.z) < 1e-12


def test_inverse_alias():
    q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 4)
    assert q.inverse().w == q.conjugate().w
    assert q.inverse().z == q.conjugate().z


# ---------- rotation semantics ----------------------------------------


def test_rotate_90_around_x_maps_y_to_z():
    q = Quaternion.from_axis_angle([1, 0, 0], math.pi / 2)
    rotated = q.rotate([0, 1, 0])
    assert abs(rotated[0]) < 1e-12
    assert abs(rotated[1]) < 1e-12
    assert abs(rotated[2] - 1.0) < 1e-12


def test_rotate_around_axis_preserves_axis():
    q = Quaternion.from_axis_angle([1, 0, 0], math.pi / 3)
    rotated = q.rotate([1.0, 0.0, 0.0])
    assert abs(rotated[0] - 1.0) < 1e-12
    assert abs(rotated[1]) < 1e-12
    assert abs(rotated[2]) < 1e-12


def test_rotate_bad_shape_raises():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    with pytest.raises(KuantShapeError):
        q.rotate([1.0, 2.0])


# ---------- axis-angle round-trip -------------------------------------


def test_axis_angle_roundtrip():
    axis_in = np.array([0.3, 0.7, 0.2])
    axis_in /= np.linalg.norm(axis_in)
    angle_in = math.pi / 2.5
    q = Quaternion.from_axis_angle(axis_in, angle_in)
    axis_out, angle_out = q.to_axis_angle()
    assert abs(angle_out - angle_in) < 1e-12
    # Axis may flip sign (equivalent representation).
    assert abs(abs(np.dot(axis_out, axis_in)) - 1.0) < 1e-12


def test_zero_angle_axis_defaults_to_x():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    axis, angle = q.to_axis_angle()
    assert angle == 0.0
    assert axis[0] == 1.0


def test_from_axis_angle_rejects_zero_axis():
    with pytest.raises(KuantValueError):
        Quaternion.from_axis_angle([0.0, 0.0, 0.0], math.pi / 4)


def test_from_axis_angle_rejects_bad_axis_shape():
    with pytest.raises(KuantShapeError):
        Quaternion.from_axis_angle([1.0, 2.0], math.pi / 4)


# ---------- rotation matrix round-trip --------------------------------


def test_rotation_matrix_agrees_with_rotate():
    q = Quaternion.from_axis_angle([1, 1, 0], math.pi / 3)
    v = np.array([1.0, 2.0, 3.0])
    R = q.to_rotation_matrix()
    v_by_R = R @ v
    v_by_q = q.rotate(v)
    np.testing.assert_allclose(v_by_R, v_by_q, atol=1e-12)


def test_from_rotation_matrix_roundtrip():
    q_in = Quaternion.from_axis_angle([0.3, 0.7, 0.2], math.pi / 4)
    R = q_in.to_rotation_matrix()
    q_out = Quaternion.from_rotation_matrix(R)
    # Roundtrip through R may flip global sign of q; check angular dist.
    d = quaternion_distance(q_in.as_array(), q_out.as_array())
    assert d < 1e-10


def test_from_rotation_matrix_rejects_bad_shape():
    with pytest.raises(KuantShapeError):
        Quaternion.from_rotation_matrix(np.eye(4))


# ---------- summary render (smoke) ------------------------------------


def test_summary_contains_angle_line():
    q = Quaternion.from_axis_angle([1, 0, 0], math.pi / 6)
    s = q.summary()
    assert "Quaternion" in s
    assert "angle" in s


# ---------- module-level array ops -----------------------------------


def test_quat_multiply_batched():
    q1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # (2, 4)
    q2 = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    out = quat_multiply(q1, q2)
    assert out.shape == (2, 4)
    # First row: identity * identity = identity.
    np.testing.assert_allclose(out[0], [1, 0, 0, 0], atol=1e-12)
    # Second row: (0, 1, 0, 0) * identity = (0, 1, 0, 0).
    np.testing.assert_allclose(out[1], [0, 1, 0, 0], atol=1e-12)


def test_quat_conjugate_flips_vector_part():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    c = quat_conjugate(q)
    np.testing.assert_allclose(c, [0.5, -0.5, -0.5, -0.5])


def test_quat_normalize_scales_to_unit():
    out = quat_normalize(np.array([2.0, 0.0, 0.0, 0.0]))
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12


def test_quat_normalize_rejects_zero_norm():
    with pytest.raises(KuantValueError):
        quat_normalize(np.array([0.0, 0.0, 0.0, 0.0]))


def test_quat_angle_identity_is_zero():
    assert abs(float(quat_angle(np.array([1.0, 0.0, 0.0, 0.0])))) < 1e-12


def test_quat_angle_matches_axis_angle():
    q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 3).as_array()
    assert abs(float(quat_angle(q)) - math.pi / 3) < 1e-12


def test_bad_last_axis_length_raises():
    with pytest.raises(KuantShapeError, match="KE-SHAPE-EXPECTED"):
        quat_multiply(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))


# ---------- slerp -----------------------------------------------------


def test_slerp_endpoints():
    q1 = Quaternion.from_axis_angle([0, 0, 1], 0.0).as_array()
    q2 = Quaternion.from_axis_angle([0, 0, 1], math.pi / 2).as_array()
    np.testing.assert_allclose(slerp(q1, q2, 0.0), q1, atol=1e-12)
    np.testing.assert_allclose(slerp(q1, q2, 1.0), q2, atol=1e-12)


def test_slerp_midpoint_half_angle():
    q1 = Quaternion.from_axis_angle([0, 0, 1], 0.0).as_array()
    q2 = Quaternion.from_axis_angle([0, 0, 1], math.pi).as_array()
    mid = slerp(q1, q2, 0.5)
    expected = Quaternion.from_axis_angle([0, 0, 1], math.pi / 2).as_array()
    # Angular distance around 0 amplifies FP error via arccos slope;
    # 1e-6 is realistic.
    d = quaternion_distance(mid, expected)
    assert d < 1e-6


def test_slerp_short_way_around():
    """When q1 . q2 < 0 the interpolation should take the short way."""
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([-1.0, 0.0, 0.0, 0.0])  # antipodal, same rotation
    mid = slerp(q1, q2, 0.5)
    # Both represent identity rotation, so mid should also be identity.
    assert abs(quat_angle(mid)) < 1e-6


def test_slerp_close_quaternions_falls_back_to_lerp():
    q1 = Quaternion.from_axis_angle([0, 0, 1], 0.0).as_array()
    q2 = Quaternion.from_axis_angle([0, 0, 1], 1e-6).as_array()
    mid = slerp(q1, q2, 0.5)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-9


# ---------- quaternion_distance --------------------------------------


def test_quaternion_distance_identical_is_zero():
    q = Quaternion.from_axis_angle([1, 0, 0], math.pi / 4).as_array()
    assert abs(float(quaternion_distance(q, q))) < 1e-12


def test_quaternion_distance_antipodal_is_zero():
    """Antipodal quaternions represent the SAME rotation."""
    q = Quaternion.from_axis_angle([1, 0, 0], math.pi / 4).as_array()
    assert abs(float(quaternion_distance(q, -q))) < 1e-12


def test_quaternion_distance_orthogonal_is_pi():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_180 = Quaternion.from_axis_angle([0, 0, 1], math.pi).as_array()
    d = float(quaternion_distance(q_id, q_180))
    assert abs(d - math.pi) < 1e-10


def test_quaternion_distance_batched():
    q1 = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    q2 = np.stack(
        [
            Quaternion.from_axis_angle([1, 0, 0], math.pi / 4).as_array(),
            Quaternion.from_axis_angle([1, 0, 0], math.pi / 2).as_array(),
        ]
    )
    d = quaternion_distance(q1, q2)
    assert d.shape == (2,)
    assert abs(float(d[0]) - math.pi / 4) < 1e-10
    assert abs(float(d[1]) - math.pi / 2) < 1e-10

"""Quaternion primitives: dataclass, Hamilton product, slerp, distance.

Convention: Hamilton (w-real-first). A unit quaternion `(w, x, y, z)`
represents a rotation by angle `2 * arccos(w)` around axis
`(x, y, z) / sin(angle/2)`. The identity rotation is `(1, 0, 0, 0)`.

Scalar and batched array support: `Quaternion` is the scalar dataclass,
handy for one-off rotations and clarity in tests. The module-level
functions `quat_multiply`, `quat_conjugate`, `quat_normalize`,
`quat_angle` operate on `(4,)` arrays or `(..., 4)` batches with numpy
broadcasting. `composerotations` and `rollholonomy` use the array form.

Design: docs/kernels/qm/quaternion/quaternion.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kuant.errors import KuantShapeError, KuantValueError

_NORM_TOL = 1e-12
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


# ---------- module-level array primitives ------------------------------


def _as_quat_array(q, *, name: str, kernel: str) -> np.ndarray:
    """Coerce input to a float64 array whose last axis is length 4."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape[-1] != 4:
        raise KuantShapeError(
            f"kuant.{kernel}: '{name}' must have last-axis length 4 "
            f"(Hamilton w, x, y, z); got shape {arr.shape}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pack as (w, x, y, z) or as (..., 4) for a batch"
        )
    return arr


def quat_multiply(q1, q2) -> np.ndarray:
    """Hamilton product `q1 * q2`, batched via numpy broadcasting."""
    a = _as_quat_array(q1, name="q1", kernel="quat_multiply")
    b = _as_quat_array(q2, name="q2", kernel="quat_multiply")
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_conjugate(q) -> np.ndarray:
    """Return `(w, -x, -y, -z)`. For a unit quaternion, this is the inverse."""
    arr = _as_quat_array(q, name="q", kernel="quat_conjugate")
    signs = np.array([1.0, -1.0, -1.0, -1.0])
    return arr * signs


def quat_normalize(q) -> np.ndarray:
    """Rescale to unit norm along the last axis. Raises on zero norm."""
    arr = _as_quat_array(q, name="q", kernel="quat_normalize")
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if bool((norm < _NORM_TOL).any()):
        raise KuantValueError(
            "kuant.quat_normalize: at least one quaternion has zero "
            "norm; the rotation it represents is undefined.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: filter out zero-norm quaternions before "
            "normalizing"
        )
    return arr / norm


def quat_angle(q) -> np.ndarray:
    """Rotation angle magnitude `2 * arccos(|w|)`, in [0, pi]."""
    arr = _as_quat_array(q, name="q", kernel="quat_angle")
    # Clip |w| to [0, 1] to defend against FP noise on nearly-unit quats.
    w = np.clip(np.abs(arr[..., 0]), 0.0, 1.0)
    return 2.0 * np.arccos(w)


def quaternion_distance(q1, q2) -> np.ndarray:
    """Angular distance between two unit quaternions, in [0, pi].

    Antipodal quaternions represent the same rotation, so we take
    `|q1 . q2|` (dot on the last axis) as the cosine of half the angle,
    then double.
    """
    a = _as_quat_array(q1, name="q1", kernel="quaternion_distance")
    b = _as_quat_array(q2, name="q2", kernel="quaternion_distance")
    dot = np.abs(np.sum(a * b, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def slerp(q1, q2, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    `t=0.0` returns `q1`, `t=1.0` returns `q2`. Values outside `[0, 1]`
    extrapolate along the great circle.

    Parameters
    ----------
    q1, q2 : (4,) array-like unit quaternions
    t : float

    Returns
    -------
    (4,) np.ndarray

    Notes
    -----
    Falls back to linear-then-normalize when the two quaternions are
    within `1e-4` of parallel (numerical safety at very small angles).
    """
    a = _as_quat_array(q1, name="q1", kernel="slerp").copy()
    b = _as_quat_array(q2, name="q2", kernel="slerp").copy()
    if a.ndim != 1 or b.ndim != 1:
        raise KuantShapeError(
            f"kuant.slerp: 'q1' and 'q2' must be 1D (4,), got shapes "
            f"{a.shape} and {b.shape}.  [KE-SHAPE-EXPECTED]"
        )
    # Choose the near hemisphere so we interpolate the short way around.
    dot = float(np.sum(a * b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 1.0 - 1e-4:
        # Nearly identical: linear + normalize is safe and avoids the
        # sin(theta) division by zero.
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    return w1 * a + w2 * b


# ---------- Quaternion dataclass ---------------------------------------


@dataclass(frozen=True)
class Quaternion:
    """Unit quaternion (Hamilton, w-first) representing a 3D rotation.

    Construction auto-normalizes any input whose norm is not exactly 1
    within `1e-12`. Zero-norm inputs raise `KuantValueError`.

    Attributes
    ----------
    w, x, y, z : float
        Component values. After construction, `w**2 + x**2 + y**2 +
        z**2 == 1` to within FP tolerance.

    Examples
    --------
    >>> import math
    >>> q_identity = Quaternion(1.0, 0.0, 0.0, 0.0)
    >>> abs(q_identity.angle() - 0.0) < 1e-12
    True
    >>> # 90 degrees around x-axis:
    >>> q = Quaternion.from_axis_angle([1.0, 0.0, 0.0], math.pi / 2)
    >>> abs(q.angle() - math.pi / 2) < 1e-12
    True
    """

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        norm_sq = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
        if norm_sq < _NORM_TOL:
            raise KuantValueError(
                f"kuant.Quaternion: input has zero norm "
                f"({norm_sq:.3g}); the rotation is undefined.  "
                f"[KE-VAL-POSITIVE]\n"
                f"  → Fix: provide a non-zero quaternion; use "
                f"Quaternion(1, 0, 0, 0) for the identity rotation"
            )
        if abs(norm_sq - 1.0) > _NORM_TOL:
            norm = math.sqrt(norm_sq)
            object.__setattr__(self, "w", self.w / norm)
            object.__setattr__(self, "x", self.x / norm)
            object.__setattr__(self, "y", self.y / norm)
            object.__setattr__(self, "z", self.z / norm)

    # ---------- accessors ---------------------------------------------

    def as_array(self) -> np.ndarray:
        """Return `(4,)` float64 array in Hamilton order."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr) -> "Quaternion":
        a = np.asarray(arr, dtype=np.float64).ravel()
        if a.size != 4:
            raise KuantShapeError(
                f"kuant.Quaternion.from_array: expected 4 elements, "
                f"got shape {np.asarray(arr).shape}.  "
                f"[KE-SHAPE-EXPECTED]"
            )
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    # ---------- algebra -----------------------------------------------

    def multiply(self, other: "Quaternion") -> "Quaternion":
        """Hamilton product `self * other`. Non-commutative."""
        return Quaternion.from_array(quat_multiply(self.as_array(), other.as_array()))

    def conjugate(self) -> "Quaternion":
        """`(w, -x, -y, -z)`. Equals inverse for unit quaternions."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quaternion":
        """Multiplicative inverse. For unit quaternions this is the conjugate."""
        # Since __post_init__ enforces unit norm, inverse == conjugate.
        return self.conjugate()

    # ---------- rotations ---------------------------------------------

    def rotate(self, v) -> np.ndarray:
        """Apply the rotation to a 3-vector `v`. Returns a length-3 array."""
        vec = np.asarray(v, dtype=np.float64).ravel()
        if vec.size != 3:
            raise KuantShapeError(
                f"kuant.Quaternion.rotate: 'v' must have 3 elements, "
                f"got shape {np.asarray(v).shape}.  "
                f"[KE-SHAPE-EXPECTED]"
            )
        # q * (0, v) * q^-1
        v_quat = np.array([0.0, vec[0], vec[1], vec[2]])
        rotated = quat_multiply(quat_multiply(self.as_array(), v_quat), self.inverse().as_array())
        return rotated[1:]

    def to_axis_angle(self) -> tuple[np.ndarray, float]:
        """Return `(axis, angle)` where axis is a length-3 unit vector."""
        angle = 2.0 * math.acos(max(-1.0, min(1.0, self.w)))
        sin_half = math.sqrt(max(0.0, 1.0 - self.w * self.w))
        if sin_half < 1e-12:
            # Zero rotation; any axis is fine. Pick x.
            return np.array([1.0, 0.0, 0.0]), 0.0
        return np.array([self.x, self.y, self.z]) / sin_half, angle

    @classmethod
    def from_axis_angle(cls, axis, angle: float) -> "Quaternion":
        """Construct from a rotation axis and angle (radians)."""
        ax = np.asarray(axis, dtype=np.float64).ravel()
        if ax.size != 3:
            raise KuantShapeError(
                f"kuant.Quaternion.from_axis_angle: 'axis' must have 3 "
                f"elements, got shape {np.asarray(axis).shape}.  "
                f"[KE-SHAPE-EXPECTED]"
            )
        norm = float(np.linalg.norm(ax))
        if norm < _NORM_TOL:
            raise KuantValueError(
                "kuant.Quaternion.from_axis_angle: 'axis' has zero norm; "
                "the rotation is undefined.  [KE-VAL-POSITIVE]\n"
                "  → Fix: pass a non-zero axis vector"
            )
        unit_ax = ax / norm
        half = 0.5 * float(angle)
        s = math.sin(half)
        return cls(math.cos(half), unit_ax[0] * s, unit_ax[1] * s, unit_ax[2] * s)

    def to_rotation_matrix(self) -> np.ndarray:
        """Return the 3x3 rotation matrix `R` such that `R @ v == self.rotate(v)`."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    @classmethod
    def from_rotation_matrix(cls, R) -> "Quaternion":
        """Invert the rotation-matrix representation. Assumes a proper R (det +1)."""
        M = np.asarray(R, dtype=np.float64)
        if M.shape != (3, 3):
            raise KuantShapeError(
                f"kuant.Quaternion.from_rotation_matrix: 'R' must be "
                f"3x3, got shape {M.shape}.  [KE-SHAPE-EXPECTED]"
            )
        tr = float(M[0, 0] + M[1, 1] + M[2, 2])
        if tr > 0.0:
            s = 2.0 * math.sqrt(1.0 + tr)
            w = 0.25 * s
            x = (M[2, 1] - M[1, 2]) / s
            y = (M[0, 2] - M[2, 0]) / s
            z = (M[1, 0] - M[0, 1]) / s
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = 2.0 * math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2])
            w = (M[2, 1] - M[1, 2]) / s
            x = 0.25 * s
            y = (M[0, 1] + M[1, 0]) / s
            z = (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = 2.0 * math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2])
            w = (M[0, 2] - M[2, 0]) / s
            x = (M[0, 1] + M[1, 0]) / s
            y = 0.25 * s
            z = (M[1, 2] + M[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1])
            w = (M[1, 0] - M[0, 1]) / s
            x = (M[0, 2] + M[2, 0]) / s
            y = (M[1, 2] + M[2, 1]) / s
            z = 0.25 * s
        return cls(float(w), float(x), float(y), float(z))

    # ---------- convenience -------------------------------------------

    def angle(self) -> float:
        """Rotation angle magnitude, in radians. Positive."""
        return 2.0 * math.acos(max(0.0, min(1.0, abs(self.w))))

    def summary(self) -> str:
        return (
            "=== Quaternion ===\n"
            f"w:       {self.w:+.6f}\n"
            f"x:       {self.x:+.6f}\n"
            f"y:       {self.y:+.6f}\n"
            f"z:       {self.z:+.6f}\n"
            f"angle:   {self.angle():.6f} rad"
        )


__all__ = [
    "Quaternion",
    "quat_multiply",
    "quat_conjugate",
    "quat_normalize",
    "quat_angle",
    "quaternion_distance",
    "slerp",
]

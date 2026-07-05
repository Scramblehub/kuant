# quaternion - Quaternion dataclass and array-level ops

## Purpose

Provide the scalar `Quaternion` primitive and the batched module-level
functions that every other kernel in this subpackage builds on. Both
layers use Hamilton (w-real-first) unit quaternions. See
[README](README.md) for how the pieces fit together.

## Public API

```python
from kuant.qm.quaternion import (
    Quaternion,
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_angle,
    quaternion_distance,
    slerp,
)
```

## `Quaternion` dataclass

Frozen dataclass, one instance per rotation. Fields `w`, `x`, `y`,
`z` are floats and always satisfy `w**2 + x**2 + y**2 + z**2 == 1`
to within `1e-12`.

```python
Quaternion(w: float, x: float, y: float, z: float)
```

### Strict unit-norm enforcement

`__post_init__` inspects the input norm.

- If `norm**2 < 1e-12`, construction raises
  `KuantValueError` with code `KE-VAL-POSITIVE`. The rotation is
  undefined and the caller must supply a non-zero quaternion (the
  identity is `Quaternion(1, 0, 0, 0)`).
- If `abs(norm**2 - 1.0) > 1e-12`, the four components are
  rescaled to unit norm in place. Non-unit input is silently
  normalized rather than rejected; the caller is expected to know
  they are constructing a rotation.

Both behaviors are covered by tests
(`test_auto_normalizes_non_unit_input`, `test_rejects_zero_norm`).

### Methods

| Method | Purpose |
| --- | --- |
| `as_array()` | Return `(4,)` float64 array `[w, x, y, z]`. |
| `from_array(arr)` | Class method. Build from a length-4 array-like; raises `KuantShapeError [KE-SHAPE-EXPECTED]` on bad size. |
| `multiply(other)` | Hamilton product `self * other`. Non-commutative. |
| `conjugate()` | Return `(w, -x, -y, -z)`. |
| `inverse()` | Multiplicative inverse. Aliased to `conjugate` because unit-norm is enforced. |
| `rotate(v)` | Apply the rotation to a length-3 vector `v`; returns a length-3 numpy array. Bad shape raises `KuantShapeError`. |
| `to_axis_angle()` | Return `(axis, angle)`. `axis` is a length-3 unit vector; `angle` is in radians. Zero rotation defaults `axis` to `(1, 0, 0)`. |
| `from_axis_angle(axis, angle)` | Class method. Build from a rotation axis and angle in radians. Rejects zero-norm axis and bad shapes. |
| `to_rotation_matrix()` | 3x3 numpy array `R` such that `R @ v == self.rotate(v)`. |
| `from_rotation_matrix(R)` | Class method. Invert the 3x3 representation. Assumes a proper rotation (`det R == +1`). |
| `angle()` | Rotation-angle magnitude in radians, in `[0, pi]`. |
| `summary()` | Short human-readable string. |

The `inverse` alias is a deliberate simplification. For a
non-unit quaternion the inverse is `conjugate / norm**2`, but every
constructed `Quaternion` is unit-norm, so the two agree.

## Module-level array ops

Every function in this section accepts numpy input whose last axis
has length 4. Shapes:

- `(4,)`: single quaternion.
- `(T, 4)`: batch of `T` quaternions (used by
  `composerotations` and `rollholonomy`).
- `(..., 4)`: arbitrary broadcast dims. `quat_multiply`,
  `quat_conjugate`, `quat_normalize`, `quat_angle`, and
  `quaternion_distance` all broadcast on the leading axes.

Bad shape raises `KuantShapeError` with code `KE-SHAPE-EXPECTED`
and a fix suggestion in the message.

### `quat_multiply(q1, q2) -> np.ndarray`

Hamilton product `q1 * q2`, computed componentwise. Batched via
numpy broadcasting.

### `quat_conjugate(q) -> np.ndarray`

Flips the sign of the vector part; returns `(w, -x, -y, -z)`.
For unit input this is the inverse.

### `quat_normalize(q) -> np.ndarray`

Rescales along the last axis. If any batch element has norm below
`1e-12`, raises `KuantValueError [KE-VAL-POSITIVE]`; the caller
must filter zero-norm inputs first.

### `quat_angle(q) -> np.ndarray`

Rotation-angle magnitude in radians, `2 * arccos(|w|)`. Clipped
to `[0, pi]`. Uses `|w|` because antipodal quaternions represent
the same rotation, so we take the shorter of the two candidate
angles.

### `quaternion_distance(q1, q2) -> np.ndarray`

Angular distance between two unit quaternions in `[0, pi]`.
Computed as `2 * arccos(|q1 . q2|)`, using absolute value so that
antipodal pairs (same rotation) give distance zero. Broadcasts on
the leading axes.

### `slerp(q1, q2, t) -> np.ndarray`

Spherical linear interpolation between two `(4,)` unit
quaternions. `t=0.0` returns `q1`, `t=1.0` returns `q2`. Values
outside `[0, 1]` extrapolate along the great circle.

Two safety details:

- If `q1 . q2 < 0`, we negate `q2` (same rotation, antipodal
  representative) so the interpolation takes the short way
  around.
- If `q1` and `q2` are within `1e-4` of parallel, we fall back to
  linear interpolation plus renormalize. The `sin(theta)` in the
  slerp formula would otherwise divide by zero.

Requires 1D inputs; batched slerp is not in this release.

## Worked example

Build two quaternions from axis and angle, compose them, rotate a
vector, and round-trip through a rotation matrix.

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import Quaternion
>>> q_x = Quaternion.from_axis_angle([1, 0, 0], math.pi / 2)  # 90 around x
>>> q_z = Quaternion.from_axis_angle([0, 0, 1], math.pi / 2)  # 90 around z
>>> combined = q_z.multiply(q_x)  # Hamilton: q_x applied first, then q_z
>>> # Apply to y-hat: q_x sends y to z, then q_z leaves z alone.
>>> rotated = combined.rotate([0.0, 1.0, 0.0])
>>> np.allclose(rotated, [0.0, 0.0, 1.0], atol=1e-10)
True
>>> R = q_x.to_rotation_matrix()
>>> np.allclose(R @ np.array([0.0, 1.0, 0.0]), q_x.rotate([0.0, 1.0, 0.0]))
True
>>> abs(Quaternion.from_rotation_matrix(R).angle() - math.pi / 2) < 1e-12
True

```

The last line recovers `pi / 2` from the 3x3 representation.
`from_rotation_matrix` may flip the global sign of the quaternion,
but the two sign choices represent the same rotation, so
`quaternion_distance` between input and output is zero.

## Related kernels

- [`composerotations`](composerotations.md): sequence composition of
  rotations built from these primitives.
- [`rollholonomy`](rollholonomy.md): trailing-window composition
  plus rotation-angle magnitude.

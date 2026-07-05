# rollholonomy - Rolling-window holonomy on a stream of rotations

## Purpose

For a `(T, 4)` panel of per-bar unit quaternions, return the composed
rotation over a trailing window at every bar and the scalar rotation
angle of that composition. Detects "the rotation path is curving" as
a per-bar time series. See [README](README.md) for framing.

## Public API

```python
from kuant.qm.quaternion import rollholonomy

holonomy, angle = rollholonomy(quats, window=22)
```

## Signature

```python
rollholonomy(quats, window: int)
```

### Inputs

- `quats`: `(T, 4)` numpy array-like of per-bar unit quaternions
  in Hamilton `(w, x, y, z)` order. 1D input and bad last-axis
  length both raise `KuantShapeError [KE-SHAPE-EXPECTED]`.
- `window`: positive integer trailing-window size. Validated by
  `require_positive`; zero or negative raises `KuantValueError`.

### Outputs

- `holonomy`: `(T, 4)` numpy array. Row `t` is the composed
  quaternion over the trailing window `[t - w + 1, t]`, applied in
  list order (see below). First `w - 1` rows are
  `(NaN, NaN, NaN, NaN)` (warm-up).
- `angle`: `(T,)` numpy array. `angle[t]` is the rotation-angle
  magnitude of `holonomy[t]` in radians, bounded to `[0, pi]`.
  First `w - 1` rows are `NaN`.

If `window > T`, both outputs are all-NaN and a
`KuantNumericWarning` fires with code `KW-VAL-WINDOW-EXCEEDS-DATA`.

## What holonomy means here

Holonomy is the accumulated rotation along a path. For a stream of
per-bar rotations `q_0, q_1, ..., q_{T-1}`, the holonomy over the
trailing window `[t - w + 1, t]` is the composed quaternion:

- **Identity holonomy** (angle near zero): the composed rotation
  is close to the identity. The window's per-bar rotations formed
  a closed loop, or there was effectively no net rotation.
- **Non-trivial holonomy** (angle > 0): the composed rotation is
  a real rotation. The path is curving in rotation space: the
  per-bar axes are drifting rather than cancelling.

`angle` is the scalar summary that downstream code typically
consumes. `holonomy` is exposed for callers who need the full
axis-angle information (for instance, to feed a distance or to
compose two panels).

## Application-order convention

Same convention as [`composerotations`](composerotations.md). The
composition of a window follows list order: `holonomy[t]` acts on a
vector as `quats[t - w + 1]` first, then `quats[t - w + 2]`, and so
on through `quats[t]`. Under the Hamilton product this is a
right-to-left multiplication of the window.

## Financial interpretation

Two common shapes:

- **Regime-drift signal.** If per-bar factor loadings are packaged as
  rotations of a fixed basis, `angle[t]` measures how far the
  window's composed rotation is from the identity. A stationary
  factor model implicitly claims the answer is close to zero. A
  visible run of non-trivial angle in the tail is a warning that the
  factor axes are drifting under that assumption.
- **HAR-RV path shape.** If lag-1, lag-5, and lag-22 realized-vol
  components are stacked as a three-axis rotation per bar, the
  22-bar holonomy angle captures whether the vol path is coasting
  in a fixed direction or curving. Coasting produces small angle;
  curving produces large angle.

Neither reading assumes a specific factor set or realized-vol
recipe. Both use the same scalar summary.

## NaN handling

`rollholonomy` propagates NaN with window granularity, not row
granularity.

- If any of the four components of `quats[s]` is NaN for
  `s in [t - w + 1, t]`, both `holonomy[t]` and `angle[t]` are
  NaN. The kernel does not try to interpolate around the gap.
- Concretely, one NaN row in the input contaminates that row and
  the `w - 1` subsequent rows in the output. A caller who wants
  gap-tolerant behavior should pre-fill or resample the input
  panel before calling `rollholonomy`.

Warm-up rows `[0, w - 2]` are unconditionally NaN.

## Errors

| Condition | Behavior |
| --- | --- |
| 1D input | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| Last-axis length not 4 | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| `window <= 0` | raises `KuantValueError` via `require_positive` |
| `window > T` | warns `KuantNumericWarning [KW-VAL-WINDOW-EXCEEDS-DATA]` and returns all-NaN |
| NaN inside a trailing window | that row's `holonomy` and `angle` are NaN |

## Complexity

Naive `O(T * w)` in v1: at each bar the window is re-composed from
scratch by repeated `quat_multiply` calls. Fast enough for the
`w = 22, 63, 252` windows typical in daily-vol research.

A prefix-product `O(T)` variant is queued for later. It needs some
care with numerical drift under repeated quaternion multiplication;
the prefix cancellation `Q_t = Q_{t + w} * Q_{t - 1}^-1` amplifies
rounding when consecutive bars are nearly identical rotations.
Users who need the faster path in the meantime can downsample the
input.

## Examples

### Uniform 30-degree rotations

Window of 3 composes 90 degrees. Window of 12 wraps past 360 and
returns to the identity.

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import Quaternion, rollholonomy
>>> q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 6).as_array()
>>> panel = np.tile(q, (12, 1))
>>> hol_3, ang_3 = rollholonomy(panel, window=3)
>>> bool(np.isnan(ang_3[:2]).all())
True
>>> all(abs(ang_3[t] - math.pi / 2) < 1e-10 for t in range(2, 12))
True
>>> _, ang_12 = rollholonomy(panel, window=12)
>>> float(ang_12[-1]) < 1e-10
True

```

The window-3 case has the first two rows NaN (warm-up) and every
subsequent row equal to `pi / 2` (three 30-degree rotations).
The window-12 case wraps 12 * 30 = 360 degrees back to the
identity.

### NaN propagates within the window

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import Quaternion, rollholonomy
>>> q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 6).as_array()
>>> panel = np.tile(q, (10, 1))
>>> panel[5] = np.nan
>>> _, ang = rollholonomy(panel, window=3)
>>> [bool(np.isnan(ang[t])) for t in [5, 6, 7]]
[True, True, True]
>>> bool(np.isfinite(ang[8]))
True

```

Row 5 carries the NaN. Rows 5, 6, 7 all include row 5 in their
trailing 3-bar window, so all three are NaN. Row 8's window is
rows 6, 7, 8, which is clean, so row 8 is finite.

### Axis switching produces non-trivial holonomy

Alternating small rotations around orthogonal axes do not cancel;
the composition curves.

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import Quaternion, rollholonomy
>>> q_x = Quaternion.from_axis_angle([1, 0, 0], math.pi / 6).as_array()
>>> q_y = Quaternion.from_axis_angle([0, 1, 0], math.pi / 6).as_array()
>>> panel = np.stack([q_x, q_y, q_x, q_y, q_x, q_y, q_x, q_y])
>>> _, ang = rollholonomy(panel, window=4)
>>> all(ang[t] > 0.1 for t in range(3, 8))
True

```

Every bar of the trailing 4-bar window carries a small rotation of
30 degrees, but the axes alternate between `x` and `y`. The
composition is a real rotation, and the angle magnitude sits well
above zero.

## Related kernels

- [`composerotations`](composerotations.md): the underlying
  sequence-composition kernel. `rollholonomy` is effectively
  `composerotations` applied to a trailing slice at every bar, with
  NaN and warm-up handling.
- [`quaternion.md`](quaternion.md): the array primitives
  (`quat_multiply`, `quat_angle`) this kernel is built on.
- [`kuant.qm.hmm`](../hmm.md): can consume `angle` as a discretized
  observation stream for regime-state inference.

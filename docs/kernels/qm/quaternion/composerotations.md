# composerotations - Compose a sequence of quaternion rotations

## Purpose

Take a sequence of per-bar rotations and return the single rotation
that, applied to a vector, has the same effect as applying every
element of the sequence in list order. Optionally return the running
composition alongside the final result. See [README](README.md) for
where this kernel sits in the subpackage.

## Public API

```python
from kuant.qm.quaternion import composerotations

Q_final                  = composerotations(quats)
Q_final, trajectory      = composerotations(quats, return_trajectory=True)
```

## Signature

```python
composerotations(quats, *, return_trajectory: bool = False)
```

### Inputs

`quats` accepts either shape:

- `(T, 4)` numpy array-like of Hamilton `(w, x, y, z)` quaternions.
- Non-empty `list` of `Quaternion` instances. Detected by checking
  `isinstance(quats[0], Quaternion)` and stacked internally.

Empty input raises `KuantValueError [KE-VAL-EMPTY]`; the caller
must pass at least one quaternion. The identity is
`(1, 0, 0, 0)`. A `(4,)` input raises `KuantShapeError` because it
is ambiguous with a batch of one; wrap it as `arr[None, :]` or a
one-element list.

### Outputs

Without `return_trajectory`:

- `Q_final`: `(4,)` numpy array. The composed rotation such that
  `Quaternion.from_array(Q_final).rotate(v)` gives the vector
  after `quats[0]`, then `quats[1]`, ..., then `quats[T-1]` have
  each been applied.

With `return_trajectory=True`:

- `Q_final`: as above.
- `trajectory`: `(T, 4)` numpy array. Row `t` is the running
  composition after applying `quats[0]` through `quats[t]`
  inclusive. `trajectory[-1]` equals `Q_final`.

## Application-order convention

**List order equals application order.** This is worth restating
because it is not the naive reading of the Hamilton product.

`composerotations([q_0, q_1, ..., q_{T-1}])` returns the composed
rotation that applies `q_0` first, then `q_1`, and so on. Under
the Hamilton product, that composed rotation is the right-to-left
product:

```text
Q = q_{T-1} * q_{T-2} * ... * q_1 * q_0
```

The reasoning: under the Hamilton convention, when a composite
rotation `p * q` acts on a vector `v` via `(p * q) * v * (p * q)^-1`,
the algebra shakes out so that `q` is applied first and `p` second.
So to make list order equal application order, the kernel multiplies
right to left. If you want the opposite convention (last-in-list
applied first), pass `quats[::-1]`.

The same convention feeds through to
[`rollholonomy`](rollholonomy.md).

## Errors

| Condition | Behavior |
| --- | --- |
| Empty `quats` | raises `KuantValueError [KE-VAL-EMPTY]` |
| 1D input | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| Last-axis length not 4 | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| Non-numeric input | numpy dtype coercion raises before the kernel does |

## Examples

### Two 45-degree z-rotations compose to 90 degrees

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import (
...     Quaternion, composerotations, quat_angle,
... )
>>> q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 4).as_array()
>>> Q = composerotations(np.stack([q, q]))
>>> abs(float(quat_angle(Q)) - math.pi / 2) < 1e-10
True

```

The rotation angle of the composition is `pi / 2` (90 degrees), as
expected for two 45-degree rotations around the same axis.

### Eight 45-degree rotations return to identity

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import (
...     Quaternion, composerotations, quat_angle,
... )
>>> q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 4).as_array()
>>> panel = np.tile(q, (8, 1))
>>> Q = composerotations(panel)
>>> float(quat_angle(Q)) < 1e-10
True

```

Eight quarter-turns around the z-axis is a full 360-degree
rotation, which is the identity up to global sign. The composed
angle is essentially zero.

### List of Quaternion objects

The kernel accepts either a `(T, 4)` numpy array or a list of
`Quaternion` instances. The two forms produce the same result.

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import (
...     Quaternion, composerotations, quat_angle,
... )
>>> quats_list = [
...     Quaternion.from_axis_angle([0, 0, 1], math.pi / 6)
...     for _ in range(3)
... ]
>>> Q_list = composerotations(quats_list)
>>> Q_array = composerotations(np.stack([q.as_array() for q in quats_list]))
>>> np.allclose(Q_list, Q_array, atol=1e-12)
True
>>> abs(float(quat_angle(Q_list)) - math.pi / 2) < 1e-10
True

```

Three 30-degree rotations compose to 90 degrees.

### Running composition via `return_trajectory`

```python
>>> import math
>>> import numpy as np
>>> from kuant.qm.quaternion import (
...     Quaternion, composerotations, quat_angle,
... )
>>> q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 6).as_array()
>>> panel = np.tile(q, (3, 1))
>>> Q_final, traj = composerotations(panel, return_trajectory=True)
>>> traj.shape
(3, 4)
>>> [round(float(quat_angle(traj[t])), 6) for t in range(3)]
[0.523599, 1.047198, 1.570796]

```

Row `t` is a rotation of `(t + 1) * 30` degrees. `traj[-1]`
equals `Q_final`.

## Related kernels

- [`quaternion.md`](quaternion.md): the array primitives
  (`quat_multiply`, `quat_conjugate`, `quat_angle`) this kernel is
  built on.
- [`rollholonomy`](rollholonomy.md): applies the same
  application-order convention on a trailing window per bar.

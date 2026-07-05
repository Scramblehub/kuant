# kuant.qm.quaternion

Hamilton (w-first) unit quaternions for 3D rotation, plus two
higher-level kernels for composing rotation streams and detecting
rolling holonomy on them.

## What lives in the subpackage

Three layers:

1. **`Quaternion` dataclass** ([`quaternion.md`](quaternion.md)).
   Scalar-convenience frozen dataclass. Enforces unit norm on
   construction. Carries per-instance methods for algebra
   (`multiply`, `conjugate`, `inverse`), rotation (`rotate`,
   `to_rotation_matrix`, `from_rotation_matrix`), and axis-angle
   representation.
2. **Module-level array ops** (also in [`quaternion.md`](quaternion.md)).
   Numpy functions that accept `(4,)` or `(..., 4)` inputs and
   broadcast on batches: `quat_multiply`, `quat_conjugate`,
   `quat_normalize`, `quat_angle`, `quaternion_distance`, `slerp`.
   These are what the two higher-level kernels build on.
3. **Kernels** on rotation streams:
   [`composerotations`](composerotations.md) composes a `(T, 4)`
   sequence of quaternions in list order, and
   [`rollholonomy`](rollholonomy.md) returns the trailing-window
   composition plus its rotation-angle magnitude per bar.

## Convention

Hamilton, w-real-first. A unit quaternion `(w, x, y, z)` represents a
rotation by angle `2 * arccos(w)` around axis
`(x, y, z) / sin(angle / 2)`. The identity rotation is
`(1, 0, 0, 0)`.

`composerotations([q_0, q_1, ..., q_{T-1}])` applies `q_0` first,
then `q_1`, then `q_2`, and so on. Under the Hamilton product this
is a right-to-left multiplication:

```text
Q = q_{T-1} * q_{T-2} * ... * q_1 * q_0
```

We chose list-order-equals-application-order because it matches
user intuition on rotation streams, at the cost of a small mental
detour when comparing against the raw Hamilton formula. Same
convention feeds through to `rollholonomy`.

## Why quaternions in a regime-analysis package

Two motivating shapes for the wider `kuant.qm` context:

- **Factor-loading drift.** Represent each bar's factor loadings as
  a rotation of a fixed basis. A stationary factor model implicitly
  assumes the composed rotation over any window is close to
  identity. A non-trivial rolling holonomy angle flags a regime
  whose axes are drifting under that assumption.
- **HAR-RV path shape.** Pack lag-1, lag-5, lag-22 realized-vol
  components into a three-axis rotation per bar. The 22-bar
  holonomy angle measures whether the vol path is coasting in a
  fixed direction (small holonomy) or curving (large holonomy).

Neither use case requires a bespoke factor model. Both need a way to
compose per-bar rotations, and a scalar per-bar summary of the
composition.

## Related kernels

- [`kuant.qm.hmm`](../hmm.md): discrete-observation Hidden Markov
  model. Pairs well with quaternion features: the rolling holonomy
  angle is a scalar observation stream that HMM state inference
  can consume.
- [`kuant.qm.zenoscan`](../zenoscan.md): retrain-frequency effect
  on any model's skill. Rolling-window kernels like `rollholonomy`
  are natural inputs.

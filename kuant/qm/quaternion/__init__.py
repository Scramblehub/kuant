"""kuant.qm.quaternion: unit-quaternion algebra for rotation-based signals.

Ships the Hamilton (w-first) quaternion primitive plus batched
array-level operations, `composerotations` for sequence composition,
and `rollholonomy` for rolling holonomy detection. Bundled utilities:
`slerp`, `quaternion_distance`, `quat_multiply`, `quat_conjugate`,
`quat_normalize`, `quat_angle`.

Use cases in the wider kuant.qm regime-analysis context:

- Model per-bar factor loadings as rotations of a fixed basis; a
  non-trivial rolling holonomy flags a regime whose factor axes are
  drifting under a stationary factor model.
- Package HAR-RV components (lag-1, lag-5, lag-22 volatility) as a
  three-axis rotation; the holonomy angle over 22 bars measures whether
  the vol path is coasting or curving.

Design: docs/kernels/qm/quaternion/README.md.
"""

from kuant.qm.quaternion.composerotations import composerotations
from kuant.qm.quaternion.quaternion import (
    Quaternion,
    quat_angle,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quaternion_distance,
    slerp,
)
from kuant.qm.quaternion.rollholonomy import rollholonomy

__all__ = [
    "Quaternion",
    "composerotations",
    "quat_angle",
    "quat_conjugate",
    "quat_multiply",
    "quat_normalize",
    "quaternion_distance",
    "rollholonomy",
    "slerp",
]

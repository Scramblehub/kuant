"""Rolling holonomy: composed rotation and its angular magnitude per window.

Holonomy is the accumulated rotation along a closed or open path. On a
stream of per-bar rotations `q_0, q_1, ..., q_{T-1}`, the holonomy over
a trailing window `[t - w + 1, t]` is the composed quaternion. If the
per-bar rotations form a closed loop, the holonomy is close to the
identity and its angle is near zero. A non-trivial holonomy over a
window signals that the path has non-zero curvature: the axis
distribution is drifting.

Use cases in the kuant.qm regime-analysis context:

- **Factor rotation drift.** Represent per-bar factor loadings as
  rotations of a fixed basis; a non-trivial rolling holonomy flags
  regime drift that a stationary factor model would miss.
- **HAR-RV path curvature.** Package (lag-1, lag-5, lag-22) volatility
  components as a rotation; the holonomy angle over a 22-bar window
  captures whether the vol path is coasting or curving.

Design: docs/kernels/qm/quaternion/rollholonomy.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import (
    require_positive,
    warn_window_exceeds_data,
)
from kuant.errors import KuantShapeError
from kuant.qm.quaternion.quaternion import _as_quat_array, quat_angle, quat_multiply


def rollholonomy(quats, window: int):
    """Composed quaternion (holonomy) over a trailing window per bar.

    Parameters
    ----------
    quats : (T, 4) array-like
        Per-bar unit quaternions in Hamilton (w, x, y, z) order.
    window : int
        Trailing-window size. Must be at least 1.

    Returns
    -------
    holonomy : (T, 4) np.ndarray
        Composed rotation over the trailing window in application
        order: `holonomy[t] rotate v` applies `quats[t-w+1]` first, then
        `quats[t-w+2]`, and so on through `quats[t]`. First `w - 1`
        rows are `(NaN, NaN, NaN, NaN)`.
    angle : (T,) np.ndarray
        `angle[t]` is the rotation magnitude of `holonomy[t]`, in radians
        in `[0, pi]`. First `w - 1` rows are `NaN`. Rows with any NaN in
        the trailing window also produce NaN in both outputs.

    Notes
    -----
    Naive O(T · w) implementation in v1. Fast enough for the
    `w = 22, 63, 252` windows typical in daily-vol research. A
    prefix-quaternion `O(T)` variant is queued for later; it needs
    care with numerical drift under repeated multiplication.

    Windows containing NaN return NaN. Windows before `w - 1` return
    NaN (warm-up).

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> # 10 rotations of 30 degrees each around z. Rolling window of 3
    >>> # composes 90 degrees. Rolling window of 12 wraps past 360.
    >>> from kuant.qm.quaternion import Quaternion
    >>> q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 6).as_array()
    >>> panel = np.tile(q, (10, 1))
    >>> hol, ang = rollholonomy(panel, window=3)
    >>> abs(ang[-1] - math.pi / 2) < 1e-12
    True
    """
    arr = _as_quat_array(quats, name="quats", kernel="rollholonomy")
    if arr.ndim != 2:
        raise KuantShapeError(
            f"kuant.rollholonomy: 'quats' must be 2D (T, 4), got shape "
            f"{arr.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: stack per-bar quaternions into a (T, 4) array"
        )
    require_positive(window, "window", kernel="rollholonomy", kind="int")
    T = arr.shape[0]
    w = int(window)
    holonomy = np.full((T, 4), np.nan, dtype=np.float64)
    angle = np.full(T, np.nan, dtype=np.float64)
    if w > T:
        warn_window_exceeds_data(w, T, kernel="rollholonomy")
        return holonomy, angle
    is_nan = ~np.isfinite(arr).all(axis=-1)
    # Slide the window.
    for t in range(w - 1, T):
        segment = arr[t - w + 1 : t + 1]
        if bool(is_nan[t - w + 1 : t + 1].any()):
            continue
        Q = segment[0]
        for i in range(1, w):
            # List order == application order via right-to-left product.
            Q = quat_multiply(segment[i], Q)
        holonomy[t] = Q
        angle[t] = quat_angle(Q)
    return holonomy, angle


__all__ = ["rollholonomy"]

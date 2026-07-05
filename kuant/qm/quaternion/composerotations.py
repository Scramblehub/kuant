"""Compose a sequence of rotations via left-to-right quaternion multiplication."""

from __future__ import annotations

import numpy as np

from kuant.errors import KuantShapeError, KuantValueError
from kuant.qm.quaternion.quaternion import Quaternion, _as_quat_array, quat_multiply


def composerotations(quats, *, return_trajectory: bool = False):
    """Compose a sequence of quaternions so list order equals application order.

    The composed rotation applies `quats[0]` FIRST, then `quats[1]`,
    etc. Under the Hamilton product, the corresponding formula is
    right-to-left: `Q = q_{T-1} * q_{T-2} * ... * q_1 * q_0`. This
    keeps user intuition ("first in list, first applied") while
    respecting the standard Hamilton convention that `p * q` applies
    `q` first then `p` when rotating a vector.

    Parameters
    ----------
    quats : (T, 4) array-like OR sequence of Quaternion
        The rotations to compose, in application order.
    return_trajectory : bool, default False
        If True, return the running composition through step `t` as a
        `(T, 4)` array in addition to the final composition. Row `t`
        is the rotation that applies `quats[0]` then `quats[1]` ...
        then `quats[t]`.

    Returns
    -------
    Q_final : (4,) np.ndarray
        The composed quaternion in Hamilton (w, x, y, z) order.
        `Quaternion.from_array(Q_final).rotate(v)` gives the vector
        after all `T` rotations have been applied in list order.
    trajectory : (T, 4) np.ndarray, optional
        Only returned when `return_trajectory=True`.

    Notes
    -----
    Quaternion multiplication is not commutative. If you want the
    reverse (last-in-list applied first) convention, pass
    `quats[::-1]`.

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> q1 = Quaternion.from_axis_angle([0, 0, 1], math.pi / 4).as_array()
    >>> q2 = Quaternion.from_axis_angle([0, 0, 1], math.pi / 4).as_array()
    >>> Q = composerotations(np.stack([q1, q2]))
    >>> # Two 45-degree rotations around z compose to a 90-degree rotation.
    >>> abs(2 * math.acos(abs(Q[0])) - math.pi / 2) < 1e-12
    True
    """
    if isinstance(quats, list) and quats and isinstance(quats[0], Quaternion):
        arr = np.stack([q.as_array() for q in quats])
    else:
        arr = _as_quat_array(quats, name="quats", kernel="composerotations")
    if arr.ndim != 2:
        raise KuantShapeError(
            f"kuant.composerotations: 'quats' must be 2D (T, 4) or a "
            f"list of Quaternion, got shape {arr.shape}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: stack into a (T, 4) array or pass a list of "
            f"Quaternion instances"
        )
    T = arr.shape[0]
    if T == 0:
        raise KuantValueError(
            "kuant.composerotations: 'quats' is empty; there is nothing "
            "to compose.  [KE-VAL-EMPTY]\n"
            "  → Fix: pass at least one quaternion; the identity is "
            "(1, 0, 0, 0)"
        )
    # Right-to-left multiplication: Q_t = q_t * Q_{t-1}. Under the
    # Hamilton product this makes Q_t apply the whole sequence up to
    # and including `quats[t]` in list order.
    if return_trajectory:
        traj = np.empty((T, 4), dtype=np.float64)
        traj[0] = arr[0]
        for t in range(1, T):
            traj[t] = quat_multiply(arr[t], traj[t - 1])
        return traj[-1].copy(), traj
    Q = arr[0].copy()
    for t in range(1, T):
        Q = quat_multiply(arr[t], Q)
    return Q


__all__ = ["composerotations"]

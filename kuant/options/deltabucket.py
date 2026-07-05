"""Select the option index nearest to each target delta.

Chain-selection primitive. Given a sorted list of options with known
deltas (e.g. all calls in an expiry), and one or more target deltas
(e.g. 0.25, 0.10 for skew nodes), return the INDEX of the option in
the chain whose delta is closest to each target.

Common trader usage:
  - "25-delta call" for RR/BF construction
  - "10-delta put" for tail hedge sizing
  - "50-delta" ATM proxy from delta axis instead of strike axis

The kernel is agnostic to whether the deltas are for calls or puts —
you pass the sign you want (e.g. -0.25 for a 25-delta put, +0.25 for
a 25-delta call).

Design: docs/kernels/options/deltabucket.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d, warn_kuant
from kuant.errors import KuantNumericWarning, KuantValueError

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _detect_backend(*args) -> Any:
    if cp is None:
        return np
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def deltabucket(deltas, targets):
    """Return the index in `deltas` closest to each target delta.

    Parameters
    ----------
    deltas : 1D array
        Deltas of the options in a chain (any monotone / arbitrary order).
    targets : scalar or 1D array
        Target delta(s) to match.

    Returns
    -------
    idx : scalar int or 1D array of ints
        For each target, the index into `deltas` of the closest match.
        Ties resolved by picking the LOWER index (numpy argmin default).

    Notes
    -----
    Signed match by convention: a target of +0.25 seeks the call-like
    +0.25 delta; -0.25 seeks the put-like -0.25 delta. The kernel does
    not enforce sign conventions — you pass what you want.

    Examples
    --------
    >>> import numpy as np
    >>> deltas = np.array([0.05, 0.15, 0.25, 0.50, 0.75, 0.95])
    >>> deltabucket(deltas, 0.25)
    2
    >>> deltabucket(deltas, np.array([0.10, 0.50, 0.90]))
    array([1, 3, 5])
    """
    xp = _detect_backend(deltas, targets)
    deltas_arr = xp.asarray(deltas)
    require_1d(deltas_arr, "deltas", kernel="deltabucket")
    if deltas_arr.size == 0:
        raise KuantValueError(
            "kuant.deltabucket: 'deltas' is empty; argmin has no index "
            "to return.  [KE-VAL-EMPTY]\n"
            "  → Fix: filter out empty chains upstream or return an "
            "empty result explicitly"
        )
    targets_arr = xp.asarray(targets)
    scalar_target = targets_arr.ndim == 0
    if scalar_target:
        targets_arr = targets_arr.reshape(1)

    # For each target, compute |delta - target| across all options and take argmin.
    # Shape: (n_targets, n_deltas) — broadcast subtraction.
    diff = xp.abs(deltas_arr[None, :] - targets_arr[:, None])
    idx = xp.argmin(diff, axis=1)

    # Flag matches whose gap exceeds a reasonable tolerance on the
    # delta scale (options deltas live in [-1, 1] for calls / [-1, 0]
    # for puts, so 0.05 is ~5 percentage points off the target).
    min_gap = xp.min(diff, axis=1)
    poor = min_gap > 0.05
    n_poor = int(poor.sum()) if xp is np else int(poor.sum().get())
    if n_poor > 0:
        warn_kuant(
            kernel="deltabucket",
            code="KW-NUM-NO-MATCH",
            what=(
                f"{n_poor} target(s) had no chain delta within 0.05; the "
                f"nearest available delta was chosen but does not represent "
                f"a meaningful bucket"
            ),
            fix=(
                "provide a chain that spans the target delta, or apply a "
                "max-gap filter downstream of this call"
            ),
            category=KuantNumericWarning,
        )

    if scalar_target:
        return int(idx[0])
    return idx

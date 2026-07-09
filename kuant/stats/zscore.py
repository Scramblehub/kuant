"""Rolling window z-score, batched.

zscore(x, w)[i] = (x[i] - rollmean(x, w)[i]) / rollstd(x, w)[i]

Composes rollmean and rollstd — the first "kernel of kernels" in
kuant.stats. NaN policy and shape/backend/dtype invariants inherit from
those composed kernels.

Zero-std policy: windows with rollstd == 0 (constant regions) produce
NaN. Natural division-by-zero behavior; downstream code decides whether
to substitute (e.g. `np.where(np.isnan(z), 0, z)`).

Design: docs/kernels/zscore.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rollmean import rollmean
from .rollstd import rollstd

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def zscore(x, window, ddof=1):
    """Rolling window z-score.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64.
    window : int
        Window size. Must be positive.
    ddof : int, default 1
        Delta degrees of freedom for the std computation. See `rollstd`.

    Returns
    -------
    1D array, same length, backend, and dtype as x
        z[i] = (x[i] - rollmean(x, w)[i]) / rollstd(x, w, ddof)[i].
        First w-1 entries NaN; windows with any NaN input NaN; windows
        with std == 0 NaN.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2, 3, 4, 5])
    >>> zscore(x, window=3)
    array([nan, nan,  0.,  0.,  0.])
    >>> # Each 3-element window is arithmetically increasing;
    >>> # its center is at the mean → z = 0.
    """
    # Delegate input validation to rollmean/rollstd; both raise on 2D/bad window.
    rmean = rollmean(x, window)
    rstd = rollstd(x, window, ddof=ddof)

    # Match rmean's backend for the arithmetic.
    if isinstance(rmean, _CUPY_NDARRAY):
        xp = cp
    else:
        xp = np

    # Coerce x to same backend/dtype as rmean for clean broadcast.
    if isinstance(x, _CUPY_NDARRAY):
        arr = x
    else:
        arr = np.asarray(x)
    if arr.dtype.kind in "iub":
        arr = arr.astype(np.float64)

    # Substitute 1.0 as denominator in zero-std cells to avoid RuntimeWarning
    # from the division, then explicitly mask them to NaN.
    one_scalar = xp.asarray(1.0, dtype=rstd.dtype)
    nan_scalar = xp.asarray(xp.nan, dtype=rstd.dtype)

    safe_std = xp.where(rstd > 0, rstd, one_scalar)
    z = (arr - rmean) / safe_std
    z = xp.where(rstd > 0, z, nan_scalar)

    return z

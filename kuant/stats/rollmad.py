"""Rolling median absolute deviation (MAD).

    rollmad(x, w)[i] = median(|x_j - median(window)|)  for j in window

Robust dispersion measure — median of deviations from the median.
Much less sensitive to outliers than standard deviation.

Implementation via sliding-window view + xp.median (twice, once for
the center and once for the MAD itself). O(n·w log w) compute.

NaN policy: strict-window via xp.median propagation.

Design: docs/kernels/rollmad.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d, require_positive, warn_window_exceeds_data

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _prepare_input(x):
    if isinstance(x, _CUPY_NDARRAY):
        arr = x
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        backend = cp
    else:
        arr = np.asarray(x)
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        backend = np

    require_1d(arr, "x", kernel="rollmad")

    return backend, arr, arr.dtype


def _sliding_view(xp, arr, w):
    if xp is np:
        from numpy.lib.stride_tricks import sliding_window_view

        return sliding_window_view(arr, w)
    from cupy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(arr, w)


def rollmad(x, window):
    """Rolling median absolute deviation.

    MAD is a robust dispersion measure: `median(|x - median(x)|)` over
    the window. Insensitive to outliers.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
    window : int

    Returns
    -------
    1D array, same length/backend/dtype.

    Examples
    --------
    >>> import numpy as np
    >>> rollmad(np.array([1.0, 2, 3, 100, 5]), 5)
    array([nan, nan, nan, nan,  2.])
    """
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollmad", kind="int")
    if w > n:
        warn_window_exceeds_data(w, n, kernel="rollmad")
        return xp.full(n, xp.nan, dtype=out_dtype)

    windowed = _sliding_view(xp, arr, w)  # (n-w+1, w)
    center = xp.median(windowed, axis=1, keepdims=True)  # (n-w+1, 1)
    deviations = xp.abs(windowed - center)  # (n-w+1, w)
    mad = xp.median(deviations, axis=1)  # (n-w+1,)

    result = xp.full(n, xp.nan, dtype=out_dtype)
    result[w - 1 :] = mad.astype(out_dtype, copy=False)
    return result

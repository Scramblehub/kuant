"""Rolling window min and max via sliding-window view.

Simple sibling to rollquantile — same sliding-view + reduction pattern,
but with xp.min / xp.max instead of xp.quantile.

Memory: O(n·w) view (strided-view on numpy; may materialize on cupy).
Compute: O(n·w) — one linear pass per window.

NaN policy — STRICT WINDOW: numpy's min/max propagate NaN by default,
so any NaN in the window produces NaN output. Matches the rest of
kuant.stats.

Design: docs/kernels/rollminmax.md.
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

    require_1d(arr, "x", kernel="rollminmax")

    return backend, arr, arr.dtype


def _sliding_view(xp, arr, w):
    if xp is np:
        from numpy.lib.stride_tricks import sliding_window_view

        return sliding_window_view(arr, w)
    from cupy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(arr, w)


def _reduce_over_windows(x, window, reducer_name):
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollminmax", kind="int")
    if w > n:
        warn_window_exceeds_data(w, n, kernel="rollminmax")
        return xp.full(n, xp.nan, dtype=out_dtype)

    windowed = _sliding_view(xp, arr, w)
    reduced = getattr(xp, reducer_name)(windowed, axis=1)

    result = xp.full(n, xp.nan, dtype=out_dtype)
    result[w - 1 :] = reduced.astype(out_dtype, copy=False)
    return result


def rollmin(x, window):
    """Rolling window minimum.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64.
    window : int
        Window size. Must be positive.

    Returns
    -------
    1D array, same length/backend/dtype
        First w-1 entries NaN (partial window); windows with any NaN
        also produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> rollmin(np.array([3.0, 1, 4, 1, 5, 9, 2, 6]), 3)
    array([nan, nan,  1.,  1.,  1.,  1.,  2.,  2.])
    """
    return _reduce_over_windows(x, window, "min")


def rollmax(x, window):
    """Rolling window maximum.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64.
    window : int
        Window size. Must be positive.

    Returns
    -------
    1D array, same length/backend/dtype

    Examples
    --------
    >>> import numpy as np
    >>> rollmax(np.array([3.0, 1, 4, 1, 5, 9, 2, 6]), 3)
    array([nan, nan,  4.,  4.,  5.,  9.,  9.,  9.])
    """
    return _reduce_over_windows(x, window, "max")

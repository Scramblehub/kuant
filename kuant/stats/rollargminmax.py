"""Rolling window argmin and argmax — position of extreme within window.

rollargmax(x, w)[i] = index within window x[i-w+1 : i+1] where max occurred
    0    = the oldest value in the window
    w-1  = the newest value (== x[i])

To convert to absolute index in original array:
    abs_idx = i - (w - 1) + rollargmax[i]
To convert to "bars ago":
    bars_ago = (w - 1) - rollargmax[i]

Same sliding-view pattern as rollminmax; xp.argmax/argmin already return
the first index of a tie, matching numpy convention.

NaN policy — STRICT WINDOW: any NaN in window → NaN output. Explicit
row mask because argmin/argmax of an all-NaN row returns 0, not NaN.

Design: docs/kernels/rollargminmax.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d, require_positive

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

    require_1d(arr, "x", kernel="rollargminmax")

    return backend, arr, arr.dtype


def _sliding_view(xp, arr, w):
    if xp is np:
        from numpy.lib.stride_tricks import sliding_window_view

        return sliding_window_view(arr, w)
    from cupy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(arr, w)


def _arg_over_windows(x, window, reducer_name):
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollargminmax", kind="int")
    if w > n:
        return xp.full(n, xp.nan, dtype=out_dtype)

    windowed = _sliding_view(xp, arr, w)
    args = getattr(xp, reducer_name)(windowed, axis=1).astype(out_dtype)

    # argmin/argmax on all-NaN rows silently returns 0. Explicit mask.
    is_nan_row = xp.any(xp.isnan(windowed), axis=1)
    nan_scalar = xp.asarray(xp.nan, dtype=out_dtype)
    args = xp.where(is_nan_row, nan_scalar, args)

    result = xp.full(n, xp.nan, dtype=out_dtype)
    result[w - 1 :] = args
    return result


def rollargmax(x, window):
    """Rolling argmax — index within window where max occurred.

    Returns
    -------
    1D float array, same length as x. First w-1 entries NaN; windows
    with any NaN produce NaN. Otherwise integer values in [0, w-1]
    stored as floats (to allow NaN).

    Examples
    --------
    >>> import numpy as np
    >>> rollargmax(np.array([3.0, 1, 4, 1, 5, 9, 2, 6]), 3)
    array([nan, nan,  2.,  0.,  2.,  2.,  1.,  2.])
    """
    return _arg_over_windows(x, window, "argmax")


def rollargmin(x, window):
    """Rolling argmin — index within window where min occurred.

    Examples
    --------
    >>> import numpy as np
    >>> rollargmin(np.array([3.0, 1, 4, 1, 5, 9, 2, 6]), 3)
    array([nan, nan,  1.,  0.,  1.,  0.,  0.,  1.])
    """
    return _arg_over_windows(x, window, "argmin")

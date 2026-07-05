"""Rolling rank of the current value within its trailing window.

rollrank(x, w)[i] = rank of x[i] among x[i-w+1 .. i+1]

Rank convention: 1-based, average-rank for ties (matches pandas).
    - Unique min in window        → 1
    - Unique max                   → w
    - Tied with k-1 others         → average of the tied positions

Optional `pct=True`: normalize by w so output is in (0, 1].

Vectorized via sliding-window view + comparison counts:
    less  = count of window values strictly less than x[i]
    equal = count of window values equal to x[i] (includes x[i] itself)
    rank  = less + (equal + 1) / 2

No sort required. O(n·w) memory (via sliding view), O(n·w) compute.

NaN policy — STRICT WINDOW: NaN comparisons are all False, so windows
containing any NaN would produce wrong counts. We detect NaN-containing
windows explicitly and mask to NaN.

Design: docs/kernels/rollrank.md.
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

    require_1d(arr, "x", kernel="rollrank")

    return backend, arr, arr.dtype


def _sliding_view(xp, arr, w):
    if xp is np:
        from numpy.lib.stride_tricks import sliding_window_view

        return sliding_window_view(arr, w)
    from cupy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(arr, w)


def rollrank(x, window, pct=False):
    """Rolling rank of the current value within the window.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64.
    window : int
        Window size. Must be positive.
    pct : bool, default False
        If True, return rank/w (in (0, 1]). If False, return the raw
        1-based rank (in [1, w] for full windows, fractional for ties).

    Returns
    -------
    1D array, same length/backend/dtype
        First w-1 entries NaN (partial window); windows with any NaN
        also produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> rollrank(np.array([3.0, 1, 4, 1, 5]), 3)
    array([nan, nan,  3. ,  1.5,  3. ])
    >>> rollrank(np.array([3.0, 1, 4, 1, 5]), 3, pct=True)
    array([nan, nan, 1. , 0.5, 1. ])
    """
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollrank", kind="int")
    if w > n:
        warn_window_exceeds_data(w, n, kernel="rollrank")
        return xp.full(n, xp.nan, dtype=out_dtype)

    windowed = _sliding_view(xp, arr, w)  # (n-w+1, w)
    last = windowed[:, -1:]  # (n-w+1, 1), broadcast target

    less = xp.sum(windowed < last, axis=1).astype(out_dtype)
    equal = xp.sum(windowed == last, axis=1).astype(out_dtype)
    rank = less + (equal + 1) / 2

    if pct:
        rank = rank / w

    # NaN in any window → NaN output. NaN comparisons return False, so
    # counts silently under-report without this guard.
    is_nan_row = xp.any(xp.isnan(windowed), axis=1)
    nan_scalar = xp.asarray(xp.nan, dtype=out_dtype)
    rank = xp.where(is_nan_row, nan_scalar, rank)

    result = xp.full(n, xp.nan, dtype=out_dtype)
    result[w - 1 :] = rank
    return result

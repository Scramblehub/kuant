"""Rolling window sum via cumulative-sum trick.

rollsum(x, w)[i] = sum(x[i-w+1 : i+1]) for i >= w-1; NaN otherwise.

Same algorithmic pattern as rollmean — one O(n) cumsum, then differences.
Strict-window NaN policy. Included as a distinct primitive because:
  - Volume/transaction totals want the raw sum, not the average
  - Discrete counts (bar counts, event counts) are more natural as sums
  - Divides by 1, not w — avoids a per-element multiply if a caller
    wanted to bring rollmean back to a sum

Design: docs/kernels/rollsum.md.
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

    require_1d(arr, "x", kernel="rollsum")

    return backend, arr, arr.dtype


def rollsum(x, window):
    """Rolling window sum.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64.
    window : int
        Window size. Must be positive.

    Returns
    -------
    1D array, same length/backend/dtype
        First w-1 entries NaN; windows with any NaN also produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> rollsum(np.array([1.0, 2, 3, 4, 5]), 3)
    array([nan, nan,  6.,  9., 12.])
    """
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollsum", kind="int")
    if w > n:
        return xp.full(n, xp.nan, dtype=out_dtype)

    is_nan = xp.isnan(arr)
    zero_scalar = xp.asarray(0, dtype=out_dtype)
    x_safe = xp.where(is_nan, zero_scalar, arr)

    zero_pad = xp.zeros(1, dtype=out_dtype)
    csum = xp.concatenate([zero_pad, xp.cumsum(x_safe)])

    nan_int = is_nan.astype(np.int64)
    nan_zero_pad = xp.zeros(1, dtype=np.int64)
    nan_csum = xp.concatenate([nan_zero_pad, xp.cumsum(nan_int)])

    window_sum = csum[w:] - csum[:-w]
    window_nan_count = nan_csum[w:] - nan_csum[:-w]

    result = xp.full(n, xp.nan, dtype=out_dtype)
    valid = window_nan_count == 0
    result[w - 1 :] = xp.where(valid, window_sum, xp.asarray(xp.nan, dtype=out_dtype))

    return result

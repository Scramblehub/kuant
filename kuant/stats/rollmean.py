"""Rolling window mean, batched via the cumulative-sum trick.

rollmean(x, w)[i] = mean(x[i-w+1 : i+1])  for i >= w-1
                  = NaN                     for i < w-1  (partial window)

Uses the O(n) cumsum trick:
    csum[i]        = sum(x[0..i])
    window_sum[i]  = csum[i] - csum[i-w]
    rollmean[i]    = window_sum[i] / w

Naive rolling-window: O(n*w). Cumsum trick: O(n). Independent of window size.

NaN policy — STRICT WINDOW:
  If ANY value in the window is NaN, output NaN for that window. Windows
  without any NaN compute normally. Matches pandas min_periods=window.

  Implementation: cumsum on NaN-replaced-by-0 array, plus a parallel
  cumsum on the NaN indicator. Windows with count > 0 get NaN.

Design: docs/kernels/rollmean.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d, require_positive

cp: Any
try:
    import cupy as cp

    _HAS_CUPY = True
    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _HAS_CUPY = False
    _CUPY_NDARRAY = type(None)


def _prepare_input(x):
    """Coerce input into (backend, arr, out_dtype). Requires 1D array."""
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

    require_1d(arr, "x", kernel="rollmean")

    return backend, arr, arr.dtype


def rollmean(x, window):
    """Rolling window mean, O(n) via cumulative-sum trick.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64. NaN propagates strictly.
    window : int
        Window size. Must be positive. If > len(x), returns all NaN.

    Returns
    -------
    1D array, same length, backend, and dtype as x
        First w-1 entries are NaN (partial window); windows containing any
        NaN also produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> rollmean(np.array([1.0, 2, 3, 4, 5]), window=3)
    array([nan, nan,  2.,  3.,  4.])
    >>> rollmean(np.array([1.0, 2, np.nan, 4, 5]), window=3)
    array([nan, nan, nan, nan, nan])
    """
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollmean", kind="int")
    if w > n:
        return xp.full(n, xp.nan, dtype=out_dtype)

    # Replace NaN with 0 for cumsum; track NaN positions separately.
    is_nan = xp.isnan(arr)
    zero_scalar = xp.asarray(0, dtype=out_dtype)
    x_safe = xp.where(is_nan, zero_scalar, arr)

    # Prepend 0 so window_sum indexes cleanly: csum[i+w] - csum[i].
    zero_pad = xp.zeros(1, dtype=out_dtype)
    csum = xp.concatenate([zero_pad, xp.cumsum(x_safe)])

    # Same trick for NaN count.
    nan_int = is_nan.astype(np.int64)
    nan_zero_pad = xp.zeros(1, dtype=np.int64)
    nan_csum = xp.concatenate([nan_zero_pad, xp.cumsum(nan_int)])

    window_sum = csum[w:] - csum[:-w]  # len n-w+1
    window_nan_count = nan_csum[w:] - nan_csum[:-w]  # len n-w+1

    # Assemble output: first w-1 stay NaN; the rest are window_sum / w
    # unless the window contained any NaN.
    result = xp.full(n, xp.nan, dtype=out_dtype)
    valid = window_nan_count == 0
    result[w - 1 :] = xp.where(valid, window_sum / w, xp.asarray(xp.nan, dtype=out_dtype))

    return result

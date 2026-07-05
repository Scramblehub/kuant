"""Rolling covariance via shifted cumsum trick.

rollcov(x, y, w, ddof=1) = cov of x and y within trailing window.

Same math and shifting trick as rollcorr (subtract x[0], y[0] before
cumsum for stability), but returns the RAW covariance instead of
normalizing to correlation.

Design: docs/kernels/rollcov.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import (
    require_1d,
    require_equal_length,
    require_nonnegative,
    require_positive,
    warn_ddof_exceeds_window,
    warn_window_exceeds_data,
)

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _prepare_inputs(x, y):
    if isinstance(x, _CUPY_NDARRAY) or isinstance(y, _CUPY_NDARRAY):
        backend = cp
        x_arr = cp.asarray(x)
        y_arr = cp.asarray(y)
    else:
        backend = np
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

    if x_arr.dtype.kind in "iub":
        x_arr = x_arr.astype(np.float64)
    if y_arr.dtype.kind in "iub":
        y_arr = y_arr.astype(np.float64)

    require_1d(x_arr, "x", kernel="rollcov")
    require_1d(y_arr, "y", kernel="rollcov")
    require_equal_length(x_arr, "x", y_arr, "y", kernel="rollcov")

    out_dtype = backend.result_type(x_arr.dtype, y_arr.dtype)
    x_arr = x_arr.astype(out_dtype, copy=False)
    y_arr = y_arr.astype(out_dtype, copy=False)

    return backend, x_arr, y_arr, out_dtype


def rollcov(x, y, window, ddof=1):
    """Rolling covariance between two 1D series.

    Parameters
    ----------
    x, y : 1D arrays of equal length
    window : int
    ddof : int, default 1
        Sample covariance uses w-1 in the denominator (matches pandas).

    Returns
    -------
    1D array, same length/backend/dtype
        Windows with any NaN in either series produce NaN.
    """
    xp, x, y, out_dtype = _prepare_inputs(x, y)
    n = x.size
    w = int(window)

    require_positive(w, "window", kernel="rollcov", kind="int")
    require_nonnegative(ddof, "ddof", kernel="rollcov", kind="int")
    if w > n:
        warn_window_exceeds_data(w, n, kernel="rollcov")
        return xp.full(n, xp.nan, dtype=out_dtype)
    denom = w - ddof
    if denom <= 0:
        warn_ddof_exceeds_window(int(ddof), w, kernel="rollcov")
        return xp.full(n, xp.nan, dtype=out_dtype)

    is_nan = xp.isnan(x) | xp.isnan(y)
    zero_scalar = xp.asarray(0, dtype=out_dtype)
    x_safe = xp.where(is_nan, zero_scalar, x)
    y_safe = xp.where(is_nan, zero_scalar, y)

    # Shifts for numerical stability.
    shift_x_val = float(x_safe[0]) if n > 0 else 0.0
    shift_y_val = float(y_safe[0]) if n > 0 else 0.0
    if not np.isfinite(shift_x_val):
        shift_x_val = 0.0
    if not np.isfinite(shift_y_val):
        shift_y_val = 0.0
    xs = x_safe - xp.asarray(shift_x_val, dtype=out_dtype)
    ys = y_safe - xp.asarray(shift_y_val, dtype=out_dtype)

    zpad = xp.zeros(1, dtype=out_dtype)
    csx = xp.concatenate([zpad, xp.cumsum(xs)])
    csy = xp.concatenate([zpad, xp.cumsum(ys)])
    csxy = xp.concatenate([zpad, xp.cumsum(xs * ys)])

    nan_int = is_nan.astype(np.int64)
    nzpad = xp.zeros(1, dtype=np.int64)
    csnan = xp.concatenate([nzpad, xp.cumsum(nan_int)])

    sx = csx[w:] - csx[:-w]
    sy = csy[w:] - csy[:-w]
    sxy = csxy[w:] - csxy[:-w]
    nnan = csnan[w:] - csnan[:-w]

    # sum((x-mux)(y-muy)) = sum(xy) - sum(x)*sum(y)/w
    cov_num = sxy - sx * sy / w
    cov_w = cov_num / denom

    result = xp.full(n, xp.nan, dtype=out_dtype)
    valid = nnan == 0
    result[w - 1 :] = xp.where(valid, cov_w, xp.asarray(xp.nan, dtype=out_dtype))
    return result

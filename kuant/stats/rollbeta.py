"""Rolling regression coefficient (beta) of y on x.

    beta = cov(x, y) / var(x)

Fundamental for CAPM factor exposures, pairs trading hedge ratios,
and any rolling-linear-model application.

Composes rollcov and a variance-of-x cumsum. Reuses the same shifting
trick for numerical stability.

Design: docs/kernels/rollbeta.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import (
    require_1d,
    require_equal_length,
    require_positive,
    warn_window_exceeds_data,
)
from kuant.errors import KuantValueError

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

    require_1d(x_arr, "x", kernel="rollbeta")
    require_1d(y_arr, "y", kernel="rollbeta")
    require_equal_length(x_arr, "x", y_arr, "y", kernel="rollbeta")

    out_dtype = backend.result_type(x_arr.dtype, y_arr.dtype)
    return (
        backend,
        x_arr.astype(out_dtype, copy=False),
        y_arr.astype(out_dtype, copy=False),
        out_dtype,
    )


def rollbeta(x, y, window):
    """Rolling regression coefficient of y on x.

    Parameters
    ----------
    x : 1D array (explanatory / independent variable)
    y : 1D array (response / dependent variable)
    window : int

    Returns
    -------
    1D array, same length/backend/dtype
        NaN where var(x) is zero (undefined slope) or window has any NaN.

    Notes
    -----
    Slope of the OLS regression y ~ alpha + beta*x fit over each
    trailing window. Direct application: rolling CAPM beta,
    pairs-trading hedge ratio.
    """
    xp, x, y, out_dtype = _prepare_inputs(x, y)
    n = x.size
    w = int(window)

    require_positive(w, "window", kernel="rollbeta", kind="int")
    if w < 2:
        raise KuantValueError(
            f"kuant.rollbeta: 'window' must be >= 2 (both variance and "
            f"covariance are undefined for a 1-element window); got "
            f"window={w}.  [KE-VAL-RANGE]\n"
            f"  → Fix: increase window to at least 2"
        )
    if w > n:
        warn_window_exceeds_data(w, n, kernel="rollbeta")
        return xp.full(n, xp.nan, dtype=out_dtype)

    is_nan = xp.isnan(x) | xp.isnan(y)
    zero_scalar = xp.asarray(0, dtype=out_dtype)
    x_safe = xp.where(is_nan, zero_scalar, x)
    y_safe = xp.where(is_nan, zero_scalar, y)

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
    csx2 = xp.concatenate([zpad, xp.cumsum(xs * xs)])

    nan_int = is_nan.astype(np.int64)
    nzpad = xp.zeros(1, dtype=np.int64)
    csnan = xp.concatenate([nzpad, xp.cumsum(nan_int)])

    sx = csx[w:] - csx[:-w]
    sy = csy[w:] - csy[:-w]
    sxy = csxy[w:] - csxy[:-w]
    sx2 = csx2[w:] - csx2[:-w]
    nnan = csnan[w:] - csnan[:-w]

    cov_num = sxy - sx * sy / w  # unnormalized
    varx_num = sx2 - sx * sx / w  # unnormalized

    # The (w - ddof) factor cancels in the ratio, so beta doesn't need ddof.
    varx_num = xp.maximum(varx_num, zero_scalar)
    denom_safe = xp.where(varx_num > 0, varx_num, xp.asarray(1.0, dtype=out_dtype))
    beta = cov_num / denom_safe
    beta = xp.where(varx_num > 0, beta, xp.asarray(xp.nan, dtype=out_dtype))

    result = xp.full(n, xp.nan, dtype=out_dtype)
    valid = nnan == 0
    result[w - 1 :] = xp.where(valid, beta, xp.asarray(xp.nan, dtype=out_dtype))
    return result

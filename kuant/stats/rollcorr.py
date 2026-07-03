"""Rolling window Pearson correlation, batched via shifted cumsum trick.

rollcorr(x, y, w)[i] = corr(x[i-w+1..i+1], y[i-w+1..i+1])
                    = cov(x, y) / (std(x) * std(y))     over the window

Math for a window of size w:
    cov  = sum(x*y) - sum(x)*sum(y)/w    (unnormalized)
    varx = sum(x²)  - sum(x)²/w          (unnormalized)
    vary = sum(y²)  - sum(y)²/w          (unnormalized)
    corr = cov / sqrt(varx * vary)

The (w - ddof) factor cancels out of the ratio, so rollcorr takes no ddof.

Shifted cumsum trick (same as rollstd): subtract x[0] and y[0] before the
cumsums to keep magnitudes small and avoid catastrophic cancellation.
Correlation is invariant under both shift AND positive scale, so the answer
is unchanged.

NaN policy — STRICT WINDOW UNION: if either x or y has any NaN in the
window, output NaN. Runs one shared NaN indicator over the union
(is_nan_x | is_nan_y).

Result clipped to [-1, 1] to defend against floating-point noise pushing
just past the bounds.

Design: docs/kernels/rollcorr.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d, require_equal_length, require_positive

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _prepare_inputs(x, y):
    """Coerce (x, y) into (backend, x_arr, y_arr, out_dtype). Both 1D, equal length."""
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

    require_1d(x_arr, "x", kernel="rollcorr")
    require_1d(y_arr, "y", kernel="rollcorr")
    require_equal_length(x_arr, "x", y_arr, "y", kernel="rollcorr")

    out_dtype = backend.result_type(x_arr.dtype, y_arr.dtype)
    x_arr = x_arr.astype(out_dtype, copy=False)
    y_arr = y_arr.astype(out_dtype, copy=False)

    return backend, x_arr, y_arr, out_dtype


def rollcorr(x, y, window):
    """Rolling Pearson correlation between two 1D series.

    Parameters
    ----------
    x, y : 1D arrays of equal length (numpy or cupy)
        Input series. Ints promote to float64.
    window : int
        Window size. Must be positive.

    Returns
    -------
    1D array, same length/backend/dtype
        Rolling correlation in [-1, 1]. First w-1 entries NaN;
        windows with any NaN in x OR y also produce NaN; windows with
        zero variance in either series produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2, 3, 4, 5])
    >>> y = np.array([2.0, 4, 6, 8, 10])  # perfectly correlated
    >>> rollcorr(x, y, 3)
    array([nan, nan,  1.,  1.,  1.])
    """
    xp, x, y, out_dtype = _prepare_inputs(x, y)
    n = x.size
    w = int(window)

    require_positive(w, "window", kernel="rollcorr", kind="int")
    if w > n:
        return xp.full(n, xp.nan, dtype=out_dtype)
    if w < 2:
        # No dispersion in a single-element window; correlation undefined.
        return xp.full(n, xp.nan, dtype=out_dtype)

    # Union NaN mask.
    is_nan = xp.isnan(x) | xp.isnan(y)
    zero_scalar = xp.asarray(0, dtype=out_dtype)
    x_safe = xp.where(is_nan, zero_scalar, x)
    y_safe = xp.where(is_nan, zero_scalar, y)

    # Shifts for numerical stability (correlation is shift-invariant).
    shift_x_val = float(x_safe[0])
    shift_y_val = float(y_safe[0])
    if not np.isfinite(shift_x_val):
        shift_x_val = 0.0
    if not np.isfinite(shift_y_val):
        shift_y_val = 0.0
    shift_x = xp.asarray(shift_x_val, dtype=out_dtype)
    shift_y = xp.asarray(shift_y_val, dtype=out_dtype)

    xs = x_safe - shift_x
    ys = y_safe - shift_y

    # Cumsums.
    zpad = xp.zeros(1, dtype=out_dtype)
    csx = xp.concatenate([zpad, xp.cumsum(xs)])
    csy = xp.concatenate([zpad, xp.cumsum(ys)])
    csxy = xp.concatenate([zpad, xp.cumsum(xs * ys)])
    csx2 = xp.concatenate([zpad, xp.cumsum(xs * xs)])
    csy2 = xp.concatenate([zpad, xp.cumsum(ys * ys)])

    nan_int = is_nan.astype(np.int64)
    nzpad = xp.zeros(1, dtype=np.int64)
    csnan = xp.concatenate([nzpad, xp.cumsum(nan_int)])

    # Window sums.
    sx = csx[w:] - csx[:-w]
    sy = csy[w:] - csy[:-w]
    sxy = csxy[w:] - csxy[:-w]
    sx2 = csx2[w:] - csx2[:-w]
    sy2 = csy2[w:] - csy2[:-w]
    nnan = csnan[w:] - csnan[:-w]

    # Unnormalized cov / var (division by w-ddof cancels in the ratio).
    cov_num = sxy - sx * sy / w
    varx_num = sx2 - sx * sx / w
    vary_num = sy2 - sy * sy / w

    # Guard tiny negatives from FP rounding.
    varx_num = xp.maximum(varx_num, zero_scalar)
    vary_num = xp.maximum(vary_num, zero_scalar)

    # corr = cov / (std_x * std_y) = cov_num / sqrt(varx_num * vary_num)
    denom = xp.sqrt(varx_num * vary_num)
    denom_safe = xp.where(denom > 0, denom, xp.asarray(1.0, dtype=out_dtype))
    corr = cov_num / denom_safe

    # Clip to [-1, 1] to defend against FP noise.
    one = xp.asarray(1.0, dtype=out_dtype)
    corr = xp.clip(corr, -one, one)

    # Zero-variance windows -> NaN.
    corr = xp.where(denom > 0, corr, xp.asarray(xp.nan, dtype=out_dtype))

    result = xp.full(n, xp.nan, dtype=out_dtype)
    valid = nnan == 0
    result[w - 1 :] = xp.where(valid, corr, xp.asarray(xp.nan, dtype=out_dtype))

    return result

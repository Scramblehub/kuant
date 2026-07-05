"""Rolling window standard deviation, via shifted cumsum trick.

rollstd(x, w, ddof=1)[i] = sqrt(sum((x_j - mu_i)²) / (w - ddof))
    for j in the window [i-w+1 .. i], mu_i = mean of that window.

Math: sum((x - mu)²) = sum(x²) - sum(x)² / w (algebraic identity).

Direct application of `E[X²] - (E[X])²` cancels badly when values are large
(e.g. S&P prices ~4000: 7 digits lost to cancellation). We defuse the
cancellation by SHIFTING: subtract `x[0]` before the cumsums, so both
sums stay small. Variance is shift-invariant, so the final answer is
correct — but numerics stay in the safe range.

NaN policy — STRICT WINDOW (matches rollmean and pandas min_periods=w).

ddof: default 1 (sample std). Set 0 for population. `w - ddof <= 0`
returns all NaN (no degrees of freedom).

Design: docs/kernels/rollstd.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import (
    require_1d,
    require_nonnegative,
    require_positive,
    warn_ddof_exceeds_window,
    warn_window_exceeds_data,
)

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
    """Coerce input into (backend, arr, out_dtype). 1D only."""
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

    require_1d(arr, "x", kernel="rollstd")

    return backend, arr, arr.dtype


def rollstd(x, window, ddof=1):
    """Rolling window standard deviation.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64.
    window : int
        Window size. Must be positive.
    ddof : int, default 1
        Delta degrees of freedom. 1 = sample std (pandas default); 0 =
        population std. If `window - ddof <= 0`, returns all NaN.

    Returns
    -------
    1D array, same length, backend, and dtype as x
        First w-1 entries are NaN (partial window); windows containing any
        NaN also produce NaN.

    Notes
    -----
    Numerical stability: uses a shifted cumsum trick. `y = x - x[0]`, then
    variance via `sum(y²) - sum(y)²/w`. Shift-invariance of variance means
    the answer equals `Var(x)` exactly, but working in the small-magnitude
    y-space avoids catastrophic cancellation.

    Precision typically matches pandas to ~1e-10; degrades slightly for
    inputs whose magnitude varies by many orders of magnitude within the
    window.

    Examples
    --------
    >>> import numpy as np
    >>> rollstd(np.array([1.0, 2, 3, 4, 5]), window=3)
    array([nan, nan,  1.,  1.,  1.])
    >>> rollstd(np.array([5.0, 5, 5, 5]), window=2)
    array([nan,  0.,  0.,  0.])
    """
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollstd", kind="int")
    require_nonnegative(ddof, "ddof", kernel="rollstd", kind="int")
    if w > n:
        warn_window_exceeds_data(w, n, kernel="rollstd")
        return xp.full(n, xp.nan, dtype=out_dtype)

    denom = w - ddof
    if denom <= 0:
        warn_ddof_exceeds_window(int(ddof), w, kernel="rollstd")
        return xp.full(n, xp.nan, dtype=out_dtype)

    is_nan = xp.isnan(arr)
    zero_scalar = xp.asarray(0, dtype=out_dtype)
    x_safe = xp.where(is_nan, zero_scalar, arr)

    # Shift for numerical stability. x_safe[0] is a "typical" value that
    # keeps y = x - shift small. If x[0] was NaN, x_safe[0] is 0 which
    # still works (just less optimal).
    shift_val = float(x_safe[0]) if n > 0 else 0.0
    if not np.isfinite(shift_val):
        shift_val = 0.0
    shift = xp.asarray(shift_val, dtype=out_dtype)

    y = x_safe - shift

    zero_pad = xp.zeros(1, dtype=out_dtype)
    csum_y = xp.concatenate([zero_pad, xp.cumsum(y)])
    csum_y_sq = xp.concatenate([zero_pad, xp.cumsum(y * y)])

    # NaN count over the window (independent of shift).
    nan_int = is_nan.astype(np.int64)
    nan_zero_pad = xp.zeros(1, dtype=np.int64)
    nan_csum = xp.concatenate([nan_zero_pad, xp.cumsum(nan_int)])

    window_sum_y = csum_y[w:] - csum_y[:-w]
    window_sum_y_sq = csum_y_sq[w:] - csum_y_sq[:-w]
    window_nan_count = nan_csum[w:] - nan_csum[:-w]

    # sum((y - y_mean)²) = sum(y²) - sum(y)² / w  (equals sum((x - x_mean)²))
    ssq = window_sum_y_sq - window_sum_y * window_sum_y / w
    # Guard tiny negatives from FP rounding.
    ssq = xp.maximum(ssq, zero_scalar)

    std_w = xp.sqrt(ssq / denom)

    result = xp.full(n, xp.nan, dtype=out_dtype)
    valid = window_nan_count == 0
    result[w - 1 :] = xp.where(valid, std_w, xp.asarray(xp.nan, dtype=out_dtype))

    return result

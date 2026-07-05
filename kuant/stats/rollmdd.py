"""Rolling maximum drawdown over a trailing window of returns.

Given a return series r, build the (local) equity curve within each
window and compute the maximum drawdown observed there:

    equity_window[j]  = ∏ (1 + r[t-w+1+k])  for k in 0..j
    peak[j]           = max(equity_window[0..j])
    dd[j]             = equity_window[j] / peak[j] - 1     (≤ 0)
    rollmdd(r, w)[t]  = min(dd[j])   over j in 0..w-1

Returned as a NEGATIVE number (e.g. -0.15 means a 15% drawdown).
NaN in warm-up and where the window contains any NaN.

O(n·w) — the drawdown depends on the whole window's shape and can't
be reduced to a cumsum trick.

Design: docs/kernels/stats/rollmdd.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_positive, warn_window_exceeds_data

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _detect_backend(x):
    if cp is not None and isinstance(x, _CUPY_NDARRAY):
        return cp
    return np


def rollmdd(x, window: int):
    """Rolling max drawdown over trailing return-windows.

    Parameters
    ----------
    x : 1D array
        Periodic returns (fractional, e.g. 0.01 for +1%).
    window : int
        Trailing window size.

    Returns
    -------
    1D array, same length as x
        Negative values (drawdowns). NaN in warm-up (first window-1) and
        where the window contains any NaN.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.array([0.05, -0.10, 0.02, -0.15, 0.03])
    >>> rollmdd(r, window=3)[-1]
    -0.15
    """
    # rollmdd has a Python-level outer loop over anchors — per-window GPU
    # dispatch overhead would dominate for reasonable window sizes. Detect
    # cupy input, drop to numpy for the computation, and transfer the
    # result back. Users still get the backend-preserving API.
    is_cupy = cp is not None and isinstance(x, _CUPY_NDARRAY)
    x_arr = np.asarray(x.get() if is_cupy else x)
    if x_arr.dtype.kind in "iub":
        x_arr = x_arr.astype(np.float64)

    n = x_arr.size
    w = int(window)
    require_positive(w, "window", kernel="rollmdd", kind="int")
    if n < w:
        warn_window_exceeds_data(w, n, kernel="rollmdd")
        out_dtype = x_arr.dtype if x_arr.dtype.kind == "f" else np.float64
        empty = np.full(n, np.nan, dtype=out_dtype)
        return cp.asarray(empty) if is_cupy else empty

    out_dtype = x_arr.dtype if x_arr.dtype.kind == "f" else np.float64
    result = np.full(n, np.nan, dtype=out_dtype)

    for t in range(w - 1, n):
        window_r = x_arr[t - w + 1 : t + 1]
        if not np.all(np.isfinite(window_r)):
            continue
        equity = np.cumprod(1.0 + window_r)
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        result[t] = np.min(dd)

    return cp.asarray(result) if is_cupy else result

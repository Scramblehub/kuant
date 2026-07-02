'''Rolling window quantile via sliding-window view.

Different algorithmic pattern from rollmean/rollstd — no cumsum trick
because quantiles don't decompose additively. Instead we build a 2D
sliding-window view of the input and take `quantile(..., axis=1)`.

Memory: O(n·w) for the view (materialized on GPU; strided-view on CPU).
Compute: O(n·w log w) for the per-window sort. For typical windows
(≤ 500) this is fast even on 100k-element inputs.

NaN policy — STRICT WINDOW: numpy's `quantile` propagates NaN by
default, so any NaN in the window produces NaN output. Same behavior
on cupy.

Design: docs/kernels/rollquantile.md.
'''
from __future__ import annotations

from typing import Any

import numpy as np

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
        if arr.dtype.kind in 'iub':
            arr = arr.astype(np.float64)
        backend = cp
    else:
        arr = np.asarray(x)
        if arr.dtype.kind in 'iub':
            arr = arr.astype(np.float64)
        backend = np

    if arr.ndim != 1:
        raise ValueError(f'rollquantile requires 1D input, got shape {arr.shape}')

    return backend, arr, arr.dtype


def _sliding_view(xp, arr, w):
    '''Return a (n-w+1, w) sliding-window view of arr. numpy: strided-view
    (zero-copy). cupy: cupyx.jit-based, may materialize.'''
    if xp is np:
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(arr, w)
    else:
        from cupy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(arr, w)


def rollquantile(x, window, q):
    '''Rolling window quantile.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values. Ints promote to float64.
    window : int
        Window size. Must be positive.
    q : float in [0, 1]
        Quantile level (0.5 = median).

    Returns
    -------
    1D array, same length/backend/dtype
        First w-1 entries NaN (partial window); windows containing any
        NaN also produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> rollquantile(np.array([1.0, 2, 3, 4, 5]), 3, 0.5)  # median
    array([nan, nan,  2.,  3.,  4.])
    >>> rollquantile(np.array([1.0, 2, 3, 4, 5]), 3, 0.25)  # 25th percentile
    array([nan, nan, 1.5, 2.5, 3.5])
    '''
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    if w <= 0:
        raise ValueError(f'window must be positive, got {w}')
    if not (0.0 <= q <= 1.0):
        raise ValueError(f'q must be in [0, 1], got {q}')
    if w > n:
        return xp.full(n, xp.nan, dtype=out_dtype)

    windowed = _sliding_view(xp, arr, w)  # (n-w+1, w)
    q_per_window = xp.quantile(windowed, q, axis=1)

    result = xp.full(n, xp.nan, dtype=out_dtype)
    result[w-1:] = q_per_window.astype(out_dtype, copy=False)

    return result


def rollmedian(x, window):
    '''Rolling window median. Convenience wrapper for `rollquantile(x, w, 0.5)`.'''
    return rollquantile(x, window, 0.5)


def rollpercentile(x, window, p):
    '''Rolling window percentile. `p` in [0, 100]. Convenience wrapper
    that calls `rollquantile(x, w, p/100)`.'''
    if not (0.0 <= p <= 100.0):
        raise ValueError(f'p must be in [0, 100], got {p}')
    return rollquantile(x, window, p / 100.0)

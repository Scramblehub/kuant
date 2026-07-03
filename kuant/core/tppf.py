'''Student-t inverse CDF (quantile function), batched.

Uses `scipy.special.stdtrit` for the numpy backend; cupy input goes
through the H↔D fallback in `_special_bridge` since `cupyx.scipy.special`
does not currently provide `stdtrit`.

Range convention: p in (0, 1). p ≤ 0 → -inf, p ≥ 1 → +inf, p out of
[0, 1] → nan.

Design: docs/kernels/core/tppf.md.
'''
from __future__ import annotations

from typing import Any

import numpy as np

from ._special_bridge import stdtrit

cp: Any
try:
    import cupy as cp
    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _detect_backend(*args) -> Any:
    if cp is None:
        return np
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def tppf(p, df):
    '''Student-t inverse CDF: return x where F(x; df) = p.

    Parameters
    ----------
    p : scalar or array
        Probability in (0, 1). Out-of-range values return sentinel
        (see below).
    df : scalar or array
        Degrees of freedom (> 0). Broadcasts with p.

    Returns
    -------
    scalar or array (float64)
        Quantile x. p = 0 → -inf, p = 1 → +inf, p ∉ [0, 1] → nan.

    Examples
    --------
    >>> abs(tppf(0.5, 5.0)) < 1e-14
    True
    >>> abs(tppf(0.975, 10.0) - 2.2281388519649425) < 1e-9
    True
    '''
    xp = _detect_backend(p, df)
    p_arr = xp.asarray(p, dtype=xp.float64)
    df_arr = xp.asarray(df, dtype=xp.float64)

    p_arr, df_arr = xp.broadcast_arrays(p_arr, df_arr)

    # Handle boundaries; stdtrit clean-region input is (0, 1).
    interior = (p_arr > 0.0) & (p_arr < 1.0)
    p_safe = xp.where(interior, p_arr, xp.asarray(0.5, dtype=xp.float64))
    df_safe = xp.where(df_arr > 0.0, df_arr, xp.asarray(1.0, dtype=xp.float64))

    result = stdtrit(df_safe, p_safe)

    neg_inf = xp.asarray(-xp.inf, dtype=xp.float64)
    pos_inf = xp.asarray(xp.inf, dtype=xp.float64)
    nan_val = xp.asarray(xp.nan, dtype=xp.float64)

    # Out of [0, 1] or invalid df → nan (checked first so -inf/+inf sentinels
    # only fire for exact 0/1).
    out_of_range = ((p_arr < 0.0) | (p_arr > 1.0) | xp.isnan(p_arr)
                     | (df_arr <= 0.0) | xp.isnan(df_arr))
    result = xp.where(p_arr == 0.0, neg_inf, result)
    result = xp.where(p_arr == 1.0, pos_inf, result)
    result = xp.where(out_of_range, nan_val, result)

    if result.ndim == 0:
        return float(result)
    return result

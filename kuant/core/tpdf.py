'''Student-t probability density function, batched.

    tpdf(x, df) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(-(ν+1)/2)

Uses log-space evaluation via `gammaln` to avoid overflow at large df.

Design: docs/kernels/core/tpdf.md.
'''
from __future__ import annotations

from typing import Any

import numpy as np

from ._special_bridge import gammaln

cp: Any
try:
    import cupy as cp
    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


_LOG_PI = 1.1447298858494002   # log(π)


def _detect_backend(*args) -> Any:
    if cp is None:
        return np
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def tpdf(x, df):
    '''Student-t PDF with `df` degrees of freedom.

    Parameters
    ----------
    x : scalar or array
        Quantile(s).
    df : scalar or array
        Degrees of freedom (> 0). Broadcasts with x.

    Returns
    -------
    scalar or array
        f(x; df) — density value(s).

    Examples
    --------
    >>> abs(tpdf(0.0, 5.0) - 0.3796066898463724) < 1e-14
    True
    '''
    xp = _detect_backend(x, df)
    x_arr = xp.asarray(x)
    df_arr = xp.asarray(df)

    out_dtype = xp.result_type(x_arr.dtype, df_arr.dtype)
    if out_dtype.kind in 'iub':
        out_dtype = xp.dtype(xp.float64)
    x_arr = x_arr.astype(out_dtype, copy=False)
    df_arr = df_arr.astype(out_dtype, copy=False)

    half = xp.asarray(0.5, dtype=out_dtype)
    df_half = half * df_arr
    df_half_plus_half = df_half + half

    # Log(pdf) = gammaln((v+1)/2) - gammaln(v/2) - 0.5 * (log(v) + log(pi))
    #           - ((v+1)/2) * log(1 + x²/v)
    log_norm = gammaln(df_half_plus_half) - gammaln(df_half) - half * (xp.log(df_arr) + _LOG_PI)
    log_body = -df_half_plus_half * xp.log1p(x_arr * x_arr / df_arr)
    result = xp.exp(log_norm + log_body)

    if result.ndim == 0:
        return float(result)
    return result

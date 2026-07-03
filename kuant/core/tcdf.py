'''Student-t cumulative distribution function, batched.

Uses the incomplete-beta identity:

    F(x; ν) = 1 - 0.5 · I_{ν/(ν+x²)}(ν/2, 1/2)     if x > 0
            = 0.5 · I_{ν/(ν+x²)}(ν/2, 1/2)         if x < 0
            = 0.5                                    if x = 0

where I_z(a, b) is the regularized incomplete beta function.

Design: docs/kernels/core/tcdf.md.
'''
from __future__ import annotations

from typing import Any

import numpy as np

from ._special_bridge import betainc

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


def tcdf(x, df):
    '''Student-t CDF: P(T <= x) where T has `df` degrees of freedom.

    Parameters
    ----------
    x : scalar or array
    df : scalar or array
        Degrees of freedom (> 0).

    Returns
    -------
    scalar or array
        F(x; df) in [0, 1].

    Examples
    --------
    >>> abs(tcdf(0.0, 5.0) - 0.5) < 1e-14
    True
    >>> abs(tcdf(2.015, 5.0) - 0.9500000000000001) < 1e-4
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

    # z = ν / (ν + x²) in [0, 1]
    z = df_arr / (df_arr + x_arr * x_arr)
    # I_z(ν/2, 1/2)
    inc_beta = betainc(half * df_arr, half, z)

    # F(x) = 0.5 · I  (x <= 0) or 1 - 0.5 · I  (x > 0). x=0 → 0.5.
    cdf_neg = half * inc_beta
    cdf_pos = 1.0 - half * inc_beta
    result = xp.where(x_arr > 0, cdf_pos, cdf_neg)
    # x exactly 0 handled by cdf_neg branch already (I_z = 1 → 0.5).

    if result.ndim == 0:
        return float(result)
    return result

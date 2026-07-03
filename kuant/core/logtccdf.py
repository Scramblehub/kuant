'''Numerically stable log of Student-t complementary CDF.

    logtccdf(x, df) = log(1 - tcdf(x, df))
                    = log(tcdf(-x, df))
                    = logtcdf(-x, df)

Trivial wrapper on `logtcdf` for readability at call sites doing
upper-tail probability calculations (fat-tail VaR, tail-loss, etc.).

Design: docs/kernels/core/logtccdf.md.
'''
from __future__ import annotations

from typing import Any

import numpy as np

from .logtcdf import logtcdf

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


def logtccdf(x, df):
    '''Log of upper-tail Student-t: `log(1 - tcdf(x, df)) = log(tcdf(-x, df))`.

    Parameters
    ----------
    x : scalar or array
    df : scalar or array

    Returns
    -------
    scalar or array (float64)
        `log(1 - tcdf(x, df))`. Finite for large positive x where naive
        `log(1 - tcdf(x))` would underflow.

    Examples
    --------
    >>> abs(logtccdf(0.0, 5.0) - -0.6931471805599453) < 1e-14
    True
    >>> import math
    >>> math.isfinite(logtccdf(1000.0, 3.0))
    True
    '''
    xp = _detect_backend(x, df)
    x_arr = xp.asarray(x)
    df_arr = xp.asarray(df)
    return logtcdf(-x_arr, df_arr)

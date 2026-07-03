'''Numerically stable log of Gaussian CDF.

    lognormcdf(x) = log(Φ(x))

Naive `log(normcdf(x))` underflows for very negative x (below ~-37 in
float64, where Φ(x) rounds to 0). Two-branch implementation:

  x >= 0:      log(1 - Φ(-x))  via  log1p(-normcdf(-x))
               — stable because Φ(-x) is small and log1p handles it
  -37 <= x < 0: log(normcdf(x))
               — normcdf(x) is representable, log is well-defined
  x < -37:     asymptotic series (Mills ratio)
               log Φ(x) ≈ -x²/2 - 0.5·log(2π) - log(-x) + log(1 - 1/x²)
               — analytic tail expansion; exact to O(1/x^4)

Design: docs/kernels/core/lognormcdf.md.
'''
from __future__ import annotations

from typing import Any

import numpy as np

from .normcdf import normcdf

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


_LOG_2PI = 1.8378770664093453   # log(2π)
_TAIL_THRESHOLD = -37.0


def lognormcdf(x):
    '''Log of Gaussian CDF: `log(Φ(x))`, numerically stable to all x.

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        Same shape as x. Float64 for int input.

    Examples
    --------
    >>> abs(lognormcdf(0.0) - -0.6931471805599453) < 1e-14
    True
    >>> # Extreme tail: naive log(normcdf(-40)) is -inf; ours is finite.
    >>> import math
    >>> math.isfinite(lognormcdf(-40.0))
    True
    '''
    xp = _detect_backend(x)
    x_arr = xp.asarray(x)
    if x_arr.dtype.kind in 'iub':
        x_arr = x_arr.astype(xp.float64)

    # Safe placeholder to avoid log(0) in inactive branches.
    x_safe = xp.where(x_arr >= _TAIL_THRESHOLD, x_arr,
                      xp.asarray(-1.0, dtype=x_arr.dtype))

    # All branches are computed vectorized; the inactive branch(es) at each
    # element may evaluate log(0) etc. but are masked out via xp.where().
    with np.errstate(divide='ignore', invalid='ignore'):
        # Branch A: x >= 0  -> log1p(-normcdf(-x))
        branch_a = xp.log1p(-normcdf(-x_safe))

        # Branch B: -37 <= x < 0  -> log(normcdf(x))
        branch_b = xp.log(normcdf(x_safe))

        # Branch C: x < -37  -> Mills asymptotic
        #   log Φ(x) ≈ -x²/2 - 0.5·log(2π) - log(-x) + log(1 - 1/x²)
        x_tail = xp.where(x_arr < _TAIL_THRESHOLD, x_arr,
                          xp.asarray(-100.0, dtype=x_arr.dtype))
        branch_c = (-0.5 * x_tail * x_tail
                    - 0.5 * _LOG_2PI
                    - xp.log(-x_tail)
                    + xp.log1p(-1.0 / (x_tail * x_tail)))

    result = xp.where(x_arr >= 0, branch_a,
                       xp.where(x_arr >= _TAIL_THRESHOLD, branch_b, branch_c))

    # NaN inputs got replaced by safe placeholders in each branch; restore.
    result = xp.where(xp.isnan(x_arr), x_arr, result)

    if result.ndim == 0:
        return float(result)
    return result

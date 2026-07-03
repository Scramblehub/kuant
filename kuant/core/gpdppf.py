'''Generalized Pareto distribution — inverse CDF (quantile function).

    x(p; ξ, σ) = σ · ((1-p)^(-ξ) - 1) / ξ                 for ξ ≠ 0
               = -σ · log(1-p)                            for ξ = 0

Domain: p in [0, 1]. Boundary conventions:
    p = 0 → 0
    p = 1 → +inf (if ξ >= 0) or -σ/ξ (upper support bound, if ξ < 0)

Design: docs/kernels/core/gpdppf.md.
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


def _detect_backend(*args) -> Any:
    if cp is None:
        return np
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def gpdppf(p, xi, scale):
    '''Generalized Pareto inverse CDF, batched.

    Parameters
    ----------
    p : scalar or array
        Probability in [0, 1].
    xi : scalar or array
        Shape parameter (ξ).
    scale : scalar or array
        Scale parameter (σ > 0).

    Returns
    -------
    scalar or array (float64)
        Quantile x satisfying gpdcdf(x; ξ, σ) = p.
        p ∉ [0, 1]: NaN.
        p = 0: 0.
        p = 1: upper support bound (+inf for ξ >= 0; -σ/ξ for ξ < 0).

    Examples
    --------
    >>> abs(gpdppf(0.5, 0.2, 1.0) - 0.7434917749851755) < 1e-14
    True
    '''
    xp = _detect_backend(p, xi, scale)
    p_arr = xp.asarray(p, dtype=xp.float64)
    xi_arr = xp.asarray(xi, dtype=xp.float64)
    scale_arr = xp.asarray(scale, dtype=xp.float64)
    p_arr, xi_arr, scale_arr = xp.broadcast_arrays(p_arr, xi_arr, scale_arr)

    # Safe placeholder to avoid log(0) or power on invalid inputs.
    interior = (p_arr > 0.0) & (p_arr < 1.0) & (scale_arr > 0)
    p_safe = xp.where(interior, p_arr, xp.asarray(0.5, dtype=xp.float64))
    one_minus_p = 1.0 - p_safe

    # ξ = 0 branch: -σ * log(1 - p)
    exp_branch = -scale_arr * xp.log(one_minus_p)

    # ξ ≠ 0 branch: σ · ((1-p)^(-ξ) - 1) / ξ
    xi_safe = xp.where(xi_arr != 0, xi_arr, xp.asarray(1.0, dtype=xp.float64))
    with np.errstate(invalid='ignore', divide='ignore'):
        pow_branch = scale_arr * (xp.power(one_minus_p, -xi_safe) - 1.0) / xi_safe

    is_zero = xp.abs(xi_arr) < 1e-8
    result = xp.where(is_zero, exp_branch, pow_branch)

    # Boundary handling.
    zero_val = xp.asarray(0.0, dtype=xp.float64)
    pos_inf = xp.asarray(xp.inf, dtype=xp.float64)
    nan_val = xp.asarray(xp.nan, dtype=xp.float64)

    # Upper support at p=1
    upper_bound = xp.where(xi_arr < 0,
                            -scale_arr / xp.where(xi_arr != 0, xi_arr,
                                                   xp.asarray(-1.0, dtype=xp.float64)),
                            pos_inf)
    result = xp.where(p_arr == 0.0, zero_val, result)
    result = xp.where(p_arr == 1.0, upper_bound, result)

    # Out of range or invalid inputs → NaN
    out_of_range = ((p_arr < 0.0) | (p_arr > 1.0) | xp.isnan(p_arr)
                     | (scale_arr <= 0) | xp.isnan(scale_arr) | xp.isnan(xi_arr))
    result = xp.where(out_of_range, nan_val, result)

    if result.ndim == 0:
        return float(result)
    return result

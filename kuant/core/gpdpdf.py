'''Generalized Pareto distribution — probability density.

    f(x; ξ, σ) = (1/σ) · (1 + ξ·x/σ)^(-1/ξ - 1)          for ξ ≠ 0
               = (1/σ) · exp(-x/σ)                        for ξ = 0

Support:
    x >= 0                                                for ξ >= 0
    0 <= x <= -σ/ξ                                        for ξ < 0

Density is 0 outside the support.

Parameters
----------
ξ (shape / tail index)  — controls tail heaviness:
    ξ > 0  → Pareto (heavy) tail — infinite variance for ξ >= 1/2
    ξ = 0  → exponential tail
    ξ < 0  → bounded support (light tail with hard upper limit)

σ (scale) > 0                                             — spread
Location μ is assumed 0; users shift x = data - μ before calling.

GPD is the LIMITING distribution of exceedances above a high threshold
(Pickands-Balkema-de Haan theorem), which is what makes it central to
Peaks-Over-Threshold (POT) tail modeling.

Design: docs/kernels/core/gpdpdf.md.
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


def gpdpdf(x, xi, scale):
    '''Generalized Pareto PDF, batched.

    Parameters
    ----------
    x : scalar or array
        Value(s).
    xi : scalar or array
        Shape parameter (ξ). Any real number.
    scale : scalar or array
        Scale parameter (σ). Must be > 0.

    Returns
    -------
    scalar or array
        Density value; 0 outside support.

    Examples
    --------
    >>> abs(gpdpdf(0.5, 0.2, 1.0) - 0.5644739300537771) < 1e-14
    True
    >>> gpdpdf(-1.0, 0.0, 1.0)   # x < 0 outside support
    0.0
    '''
    xp = _detect_backend(x, xi, scale)
    x_arr = xp.asarray(x)
    xi_arr = xp.asarray(xi)
    scale_arr = xp.asarray(scale)

    out_dtype = xp.result_type(x_arr.dtype, xi_arr.dtype, scale_arr.dtype)
    if out_dtype.kind in 'iub':
        out_dtype = xp.dtype(xp.float64)
    x_arr = x_arr.astype(out_dtype, copy=False)
    xi_arr = xi_arr.astype(out_dtype, copy=False)
    scale_arr = scale_arr.astype(out_dtype, copy=False)
    x_arr, xi_arr, scale_arr = xp.broadcast_arrays(x_arr, xi_arr, scale_arr)

    # Support: x >= 0 always; upper bound x <= -scale/xi when xi < 0.
    upper = xp.where(xi_arr < 0, -scale_arr / xp.where(xi_arr != 0, xi_arr,
                                                       xp.asarray(-1.0, dtype=out_dtype)),
                      xp.asarray(np.inf, dtype=out_dtype))
    in_support = (x_arr >= 0) & (x_arr <= upper) & (scale_arr > 0)

    # Safe placeholder for the argument to log/pow.
    y = x_arr / xp.where(scale_arr > 0, scale_arr, xp.asarray(1.0, dtype=out_dtype))
    y_safe = xp.where(in_support, y, xp.asarray(0.0, dtype=out_dtype))

    # ξ = 0 branch: exponential
    exp_branch = xp.exp(-y_safe) / xp.where(scale_arr > 0, scale_arr,
                                             xp.asarray(1.0, dtype=out_dtype))
    # ξ ≠ 0 branch: (1 + ξ·y)^(-1/ξ - 1) / σ
    xi_safe = xp.where(xi_arr != 0, xi_arr, xp.asarray(1.0, dtype=out_dtype))
    one_plus = 1.0 + xi_safe * y_safe
    # Guard against 1+ξy <= 0 in inactive cells (shouldn't happen in-support,
    # but xp.where masks it out regardless)
    with np.errstate(invalid='ignore', divide='ignore'):
        pow_branch = xp.power(xp.maximum(one_plus, 1e-300),
                              -1.0 / xi_safe - 1.0) / \
                     xp.where(scale_arr > 0, scale_arr, xp.asarray(1.0, dtype=out_dtype))

    # ξ = 0 is a distinct case (limiting exponential).
    # Numerically, ξ within 1e-8 of 0 is safer to route to the exponential branch.
    is_zero = xp.abs(xi_arr) < 1e-8
    result = xp.where(is_zero, exp_branch, pow_branch)
    result = xp.where(in_support, result, xp.asarray(0.0, dtype=out_dtype))

    if result.ndim == 0:
        return float(result)
    return result

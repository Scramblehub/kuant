"""Generalized Pareto distribution — cumulative distribution function.

    F(x; ξ, σ) = 1 - (1 + ξ·x/σ)^(-1/ξ)                   for ξ ≠ 0
               = 1 - exp(-x/σ)                            for ξ = 0

Support:
    x >= 0                                                for ξ >= 0
    0 <= x <= -σ/ξ                                        for ξ < 0

F(x) = 0 for x < 0.  F(x) = 1 above the upper support bound (ξ < 0).

Design: docs/kernels/core/gpdcdf.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant.errors import KuantValueError

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


def gpdcdf(x, xi, scale):
    """Generalized Pareto CDF, batched.

    Parameters
    ----------
    x : scalar or array
    xi : scalar or array
        Shape parameter (ξ).
    scale : scalar or array
        Scale parameter (σ > 0).

    Returns
    -------
    scalar or array
        F(x) in [0, 1].

    Examples
    --------
    >>> abs(gpdcdf(0.5, 0.2, 1.0) - 0.37907867694084507) < 1e-14
    True
    >>> gpdcdf(-1.0, 0.5, 2.0)
    0.0
    """
    xp = _detect_backend(x, xi, scale)
    x_arr = xp.asarray(x)
    xi_arr = xp.asarray(xi)
    scale_arr = xp.asarray(scale)
    if bool((xp.asarray(scale_arr).ravel() <= 0).any()):
        raise KuantValueError(
            "kuant.gpdcdf: 'scale' must be strictly positive (GPD scale σ "
            "is defined on (0, +inf)); got scale <= 0 in one or more cells."
            "  [KE-VAL-POSITIVE]\n"
            "  → Fix: pass scale > 0"
        )

    out_dtype = xp.result_type(x_arr.dtype, xi_arr.dtype, scale_arr.dtype)
    if out_dtype.kind in "iub":
        out_dtype = xp.dtype(xp.float64)
    x_arr = x_arr.astype(out_dtype, copy=False)
    xi_arr = xi_arr.astype(out_dtype, copy=False)
    scale_arr = scale_arr.astype(out_dtype, copy=False)
    x_arr, xi_arr, scale_arr = xp.broadcast_arrays(x_arr, xi_arr, scale_arr)

    y = x_arr / xp.where(scale_arr > 0, scale_arr, xp.asarray(1.0, dtype=out_dtype))
    y_safe = xp.where(x_arr >= 0, y, xp.asarray(0.0, dtype=out_dtype))

    # ξ = 0 branch: 1 - exp(-y)
    exp_branch = 1.0 - xp.exp(-y_safe)

    # ξ ≠ 0 branch: 1 - (1 + ξ·y)^(-1/ξ)
    xi_safe = xp.where(xi_arr != 0, xi_arr, xp.asarray(1.0, dtype=out_dtype))
    one_plus = 1.0 + xi_safe * y_safe
    with np.errstate(invalid="ignore", divide="ignore"):
        pow_branch = 1.0 - xp.power(xp.maximum(one_plus, 1e-300), -1.0 / xi_safe)

    is_zero = xp.abs(xi_arr) < 1e-8
    result = xp.where(is_zero, exp_branch, pow_branch)

    # Below support: 0
    result = xp.where(x_arr < 0, xp.asarray(0.0, dtype=out_dtype), result)
    # Above upper support (only when ξ < 0): 1
    upper = xp.where(
        xi_arr < 0,
        -scale_arr / xp.where(xi_arr != 0, xi_arr, xp.asarray(-1.0, dtype=out_dtype)),
        xp.asarray(np.inf, dtype=out_dtype),
    )
    result = xp.where(x_arr > upper, xp.asarray(1.0, dtype=out_dtype), result)

    if result.ndim == 0:
        return float(result)
    return result

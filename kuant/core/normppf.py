"""Inverse Gaussian CDF (percent-point function), batched.

Given probability p in (0, 1), return x such that Φ(x) = p.

Uses Peter Acklam's rational approximation (2004), accurate to ~1.15e-9
in double precision for p in (0, 1). Three regions:

    Lower tail: p < 0.02425  → uses log(p) transformation
    Central:    0.02425 <= p <= 0.97575  → uses (p - 0.5)² polynomial
    Upper tail: p > 0.97575  → uses log(1-p) transformation

Reference: https://web.archive.org/web/20150910040355/http://home.online.no/~pjacklam/notes/invnorm/

Design: docs/kernels/core/normppf.md.
"""

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


# Peter Acklam's rational approximation coefficients (double-precision).
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_PLOW = 0.02425
_PHIGH = 1.0 - _PLOW


def normppf(p):
    """Inverse standard-normal CDF: return x where Φ(x) = p.

    Parameters
    ----------
    p : scalar or array
        Probability in (0, 1). Values outside (0, 1) return NaN;
        p = 0 returns -inf, p = 1 returns +inf.

    Returns
    -------
    x : scalar or array (float64)
        Same shape as p. Accurate to ~1.15e-9 for p in (1e-300, 1-1e-16).

    Examples
    --------
    >>> normppf(0.5)
    0.0
    >>> abs(normppf(0.975) - 1.9599639845400545) < 1e-9
    True
    >>> abs(normppf(0.025) - -1.9599639845400545) < 1e-9
    True
    """
    xp = _detect_backend(p)
    p_arr = xp.asarray(p, dtype=xp.float64)

    # Region masks
    p_zero = p_arr <= 0.0
    p_one = p_arr >= 1.0
    p_nan = xp.isnan(p_arr) | (p_arr < 0.0) | (p_arr > 1.0)
    lower = (p_arr > 0.0) & (p_arr < _PLOW)
    upper = (p_arr > _PHIGH) & (p_arr < 1.0)
    central = (p_arr >= _PLOW) & (p_arr <= _PHIGH)

    # Safe placeholder to avoid log(0) / sqrt(negative)
    p_safe_lower = xp.where(lower, p_arr, 0.5)
    p_safe_upper = xp.where(upper, p_arr, 0.5)
    p_safe_central = xp.where(central, p_arr, 0.5)

    # Central region
    q_c = p_safe_central - 0.5
    r_c = q_c * q_c
    num_c = ((((_A[0] * r_c + _A[1]) * r_c + _A[2]) * r_c + _A[3]) * r_c + _A[4]) * r_c + _A[5]
    den_c = ((((_B[0] * r_c + _B[1]) * r_c + _B[2]) * r_c + _B[3]) * r_c + _B[4]) * r_c + 1.0
    x_c = num_c * q_c / den_c

    # Lower tail
    q_l = xp.sqrt(-2.0 * xp.log(p_safe_lower))
    num_l = ((((_C[0] * q_l + _C[1]) * q_l + _C[2]) * q_l + _C[3]) * q_l + _C[4]) * q_l + _C[5]
    den_l = (((_D[0] * q_l + _D[1]) * q_l + _D[2]) * q_l + _D[3]) * q_l + 1.0
    x_l = num_l / den_l

    # Upper tail — same formula on (1-p), then negate
    q_u = xp.sqrt(-2.0 * xp.log(1.0 - p_safe_upper))
    num_u = ((((_C[0] * q_u + _C[1]) * q_u + _C[2]) * q_u + _C[3]) * q_u + _C[4]) * q_u + _C[5]
    den_u = (((_D[0] * q_u + _D[1]) * q_u + _D[2]) * q_u + _D[3]) * q_u + 1.0
    x_u = -num_u / den_u

    # Assemble
    x = xp.where(central, x_c, xp.where(lower, x_l, x_u))
    neg_inf = xp.asarray(-xp.inf, dtype=xp.float64)
    pos_inf = xp.asarray(xp.inf, dtype=xp.float64)
    nan_val = xp.asarray(xp.nan, dtype=xp.float64)
    x = xp.where(p_zero, neg_inf, x)
    x = xp.where(p_one, pos_inf, x)
    x = xp.where(p_nan, nan_val, x)

    if x.ndim == 0:
        return float(x)
    return x

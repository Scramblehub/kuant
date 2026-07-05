"""Numerically stable log of Student-t CDF.

    logtcdf(x, df) = log(tcdf(x, df))

Naive `log(tcdf(x, df))` underflows to -inf for extremely negative x
where tcdf itself rounds to 0. Two-branch implementation:

  Normal range: log(tcdf(x, df))  — direct
  Underflow guard: asymptotic tail
    log(tcdf(x, df)) ≈ log(1/2) + (df/2)·log(z) - log(df/2)
                       - lnB(df/2, 1/2)
    where z = df / (df + x²) and lnB is log of the Beta function.
    Leading-order series expansion for the regularized incomplete beta
    at z → 0. Absolute error is O(z), i.e. O(1/x²) in the tail.

Design: docs/kernels/core/logtcdf.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant.errors import KuantValueError

from ._special_bridge import gammaln
from .tcdf import tcdf

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


_LOG_HALF = -0.6931471805599453  # log(1/2)
_GAMMALN_HALF = 0.5723649429247001  # gammaln(1/2) = log(sqrt(pi))


def _detect_backend(*args) -> Any:
    if cp is None:
        return np
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def logtcdf(x, df):
    """Log of Student-t CDF, numerically stable in the deep left tail.

    Parameters
    ----------
    x : scalar or array
    df : scalar or array
        Degrees of freedom (> 0). Broadcasts with x.

    Returns
    -------
    scalar or array (float64)
        `log(tcdf(x, df))`. Always finite for x with df > 0.

    Examples
    --------
    >>> abs(logtcdf(0.0, 5.0) - -0.6931471805599453) < 1e-14
    True
    >>> # Naive log(tcdf(-40, 3)) can underflow at very extreme x;
    >>> # this remains finite.
    >>> import math
    >>> math.isfinite(logtcdf(-1000.0, 3.0))
    True
    """
    xp = _detect_backend(x, df)
    x_arr = xp.asarray(x, dtype=xp.float64)
    df_arr = xp.asarray(df, dtype=xp.float64)
    if bool((df_arr.ravel() <= 0).any()):
        raise KuantValueError(
            "kuant.logtcdf: degrees of freedom 'df' must be strictly "
            "positive; got df <= 0 in one or more cells.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: pass df > 0"
        )
    x_arr, df_arr = xp.broadcast_arrays(x_arr, df_arr)

    v = tcdf(x_arr, df_arr)

    # Direct branch: log(v) works whenever v > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = xp.log(v)

    # Asymptotic branch: log(tcdf) via leading-order series of the
    # regularized incomplete beta as z → 0.
    #   log(tcdf) ≈ log(1/2) + a·log(z) - log(a) - lnB(a, 1/2)
    # where a = df/2, z = df/(df + x²), lnB(a, 1/2) = Γln(a)+Γln(1/2)-Γln(a+1/2).
    a = 0.5 * df_arr
    # Safe placeholder for the direct-branch elements (avoid log(negative)).
    x_safe = xp.where(x_arr < 0, x_arr, xp.asarray(-1.0, dtype=x_arr.dtype))
    z = df_arr / (df_arr + x_safe * x_safe)
    lnbeta_val = gammaln(a) + _GAMMALN_HALF - gammaln(a + 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        asymp = _LOG_HALF + a * xp.log(z) - xp.log(a) - lnbeta_val

    # Use asymp when v underflowed to 0 (only happens at very extreme x < 0).
    result = xp.where((x_arr < 0) & (v == 0.0), asymp, direct)
    # NaN inputs propagate.
    result = xp.where(
        xp.isnan(x_arr) | xp.isnan(df_arr) | (df_arr <= 0),
        xp.asarray(xp.nan, dtype=xp.float64),
        result,
    )

    if result.ndim == 0:
        return float(result)
    return result

"""Numerically stable log of Gaussian complementary CDF.

    lognormccdf(x) = log(1 - Φ(x))
                   = log(Φ(-x))
                   = lognormcdf(-x)

Trivial wrapper on `lognormcdf` for readability at call sites doing
tail-probability calculations.

Design: docs/kernels/core/lognormccdf.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .lognormcdf import lognormcdf

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


def lognormccdf(x):
    """Log of upper-tail Gaussian: `log(1 - Φ(x)) = log(Φ(-x))`.

    Parameters
    ----------
    x : scalar or array

    Returns
    -------
    scalar or array
        Same shape as x. Float64 for int input.

    Examples
    --------
    >>> abs(lognormccdf(0.0) - -0.6931471805599453) < 1e-14
    True
    >>> # log(1 - Φ(6)) is far below float64 log range if computed naively
    >>> import math
    >>> math.isfinite(lognormccdf(6.0))
    True
    """
    xp = _detect_backend(x)
    x_arr = xp.asarray(x)
    return lognormcdf(-x_arr)

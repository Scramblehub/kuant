"""Rolling idiosyncratic volatility — residual std after regressing y on x.

For each trailing window, fit OLS `y = alpha + beta*x + eps` and return
the standard deviation of the residuals `eps`. This is the part of y's
variability that isn't explained by x.

Elegant closed form (no per-window residual computation needed):

    var(eps) = var(y) - cov(x, y)² / var(x)
             = var(y) · (1 - corr(x, y)²)

So `rollidio(y, x, w) = sqrt(rollvar(y) · (1 - rollcorr(x, y)²))`
which composes on rollstd and rollcorr — no new cumsum work.

Direct use in kuant: idiosyncratic vol as a signal, single-name
factor-model residuals, "how much of this stock's variance is NOT
explained by SPY?"

Design: docs/kernels/rollidio.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rollcorr import rollcorr
from .rollstd import rollstd

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def rollidio(y, x, window, ddof=1):
    """Rolling idiosyncratic std: sqrt(var(y) * (1 - corr(x, y)²)).

    Parameters
    ----------
    y : 1D array
        Response / dependent series (e.g. single-name returns).
    x : 1D array of equal length
        Explanatory / factor series (e.g. SPY returns).
    window : int
    ddof : int, default 1
        Degrees of freedom for the std of y. corr is dimensionless
        so ddof doesn't affect the (1 - corr²) term.

    Returns
    -------
    1D array, same length/backend/dtype
        Idiosyncratic (residual) volatility per window.

    Notes
    -----
    Note the argument order — `y` first, `x` second — matches the
    regression semantic "y explained by x". If you swap them, you're
    computing the residual variance of x given y instead.
    """
    std_y = rollstd(y, window, ddof=ddof)
    corr_xy = rollcorr(x, y, window)

    # Determine backend from the returned std (already validated by rollstd).
    if _CUPY_NDARRAY is not type(None) and isinstance(std_y, _CUPY_NDARRAY):
        xp = cp
    else:
        xp = np

    one_minus_r2 = xp.maximum(1.0 - corr_xy * corr_xy, 0.0)
    return std_y * xp.sqrt(one_minus_r2)

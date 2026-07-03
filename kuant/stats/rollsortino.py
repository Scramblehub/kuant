"""Rolling Sortino ratio.

    downside_dev(r, w, target)[t]
        = sqrt(mean(max(target - r, 0)^2 within window))
    rollsortino(r, w, ann=1, target=rf)[t]
        = (rollmean(r, w)[t] - target) / downside_dev * sqrt(ann)

Sortino is the "downside-only Sharpe": penalizes only losses below a
minimum acceptable return (MAR / target). Reduces to Sharpe when the
target equals the mean.

Design: docs/kernels/stats/rollsortino.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rollmean import rollmean

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _detect_backend(x):
    if cp is not None and isinstance(x, _CUPY_NDARRAY):
        return cp
    return np


def rollsortino(x, window: int, ann_factor: float = 1.0, target: float = 0.0):
    """Rolling Sortino ratio, annualized.

    Parameters
    ----------
    x : 1D array
        Periodic returns.
    window : int
        Trailing window size.
    ann_factor : float, default 1.0
        Annualization factor.
    target : float, default 0.0
        Minimum Acceptable Return (MAR) per period. Returns below this
        contribute to the downside; returns above do not.

    Returns
    -------
    1D array, same length as x
        NaN in warm-up region and where downside dev is zero.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.normal(0.0005, 0.01, 1000)
    >>> s = rollsortino(r, window=252, ann_factor=252, target=0.0)
    >>> np.all(np.isnan(s[:251]))
    True
    """
    xp = _detect_backend(x)
    x_arr = xp.asarray(x)
    if x_arr.dtype.kind in "iub":
        x_arr = x_arr.astype(np.float64)

    # Downside excursion below the target.
    below = xp.where(x_arr < target, target - x_arr, xp.asarray(0.0, dtype=x_arr.dtype))
    # Mean of squared downside within window is E[(target - r)^2 | r < target] scaled by
    # (n_below / w). rollmean handles NaN strictly; use x^2 → rollmean.
    downside_sq = below * below
    down_ms = rollmean(downside_sq, window)  # mean-of-squares of downside excursions

    with np.errstate(invalid="ignore"):
        downside_dev = xp.sqrt(down_ms)

    # Numerator
    mean = rollmean(x_arr, window)
    excess = mean - target

    # NaN where downside_dev is 0 (no losses in window → Sortino undefined)
    safe_dd = xp.where(downside_dev > 0, downside_dev, xp.asarray(xp.nan, dtype=mean.dtype))
    result = excess / safe_dd
    if ann_factor != 1.0:
        result = result * xp.sqrt(xp.asarray(ann_factor, dtype=mean.dtype))
    return result

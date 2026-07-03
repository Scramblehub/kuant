"""Rolling Sharpe ratio.

    rollsharpe(r, w, ann=1)[t]
        = (rollmean(r, w)[t] - rf/ann) / rollstd(r, w)[t]   * sqrt(ann)

Sharpe = excess-return per unit volatility, annualized by sqrt(ann).
Standard convention: annual Sharpe from daily returns uses ann=252.

Design: docs/kernels/stats/rollsharpe.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rollmean import rollmean
from .rollstd import rollstd

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def rollsharpe(x, window: int, ann_factor: float = 1.0, rf: float = 0.0, ddof: int = 1):
    """Rolling Sharpe ratio, annualized.

    Parameters
    ----------
    x : 1D array
        Periodic returns (e.g. daily). Do NOT pass log returns unless
        the Sharpe formula you want is on log-returns.
    window : int
        Trailing window size.
    ann_factor : float, default 1.0
        Annualization factor. For daily returns → 252; weekly → 52;
        monthly → 12; leave at 1 for per-window Sharpe (no annualization).
    rf : float, default 0.0
        Risk-free rate PER PERIOD (not annualized). Subtracted from
        each return before the ratio.
    ddof : int, default 1
        Standard-deviation ddof (sample std by default).

    Returns
    -------
    1D array, same length as x
        NaN for windows with any NaN and for the warm-up region.
        NaN where std is zero.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.normal(0.0005, 0.01, 1000)   # daily returns
    >>> s = rollsharpe(r, window=252, ann_factor=252)
    >>> np.all(np.isnan(s[:251]))
    True
    """
    mean = rollmean(x, window)
    std = rollstd(x, window, ddof=ddof)

    xp = cp if isinstance(mean, _CUPY_NDARRAY) else np
    excess = mean - rf
    # NaN where std==0 (constant window) or std is NaN.
    safe_std = xp.where(std > 0, std, xp.asarray(xp.nan, dtype=mean.dtype))
    result = excess / safe_std
    if ann_factor != 1.0:
        result = result * xp.sqrt(xp.asarray(ann_factor, dtype=mean.dtype))
    return result

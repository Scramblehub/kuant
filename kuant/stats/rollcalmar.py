"""Rolling Calmar ratio.

    rollcalmar(r, w, ann=1)[t]
        = annualized_mean_return / |rollmdd(r, w)[t]|

Calmar penalizes strategies by their WORST recent drawdown, not
per-window volatility. Useful for tail-averse position sizing.

Design: docs/kernels/stats/rollcalmar.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import warn_kuant
from kuant.errors import KuantNumericWarning

from .rollmdd import rollmdd
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


def rollcalmar(x, window: int, ann_factor: float = 1.0):
    """Rolling Calmar ratio.

    Parameters
    ----------
    x : 1D array
        Periodic returns.
    window : int
        Trailing window.
    ann_factor : float, default 1.0
        Annualization factor. For daily returns → 252; monthly → 12.

    Returns
    -------
    1D array
        NaN in warm-up and where MDD is zero (no drawdown → undefined).

    Notes
    -----
    Signed conventions:
      - Returns are periodic (fractional, e.g. 0.01 for +1%).
      - MDD from rollmdd is negative (e.g. -0.15 for a 15% drawdown).
      - Calmar is annualized_mean / abs(MDD): always positive when
        mean return is positive; negative when mean is negative.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.normal(0.001, 0.01, 500)
    >>> c = rollcalmar(r, window=252, ann_factor=252)
    >>> np.all(np.isnan(c[:251]))
    True
    """
    xp = _detect_backend(x)
    mean = rollmean(x, window)
    mdd = rollmdd(x, window)

    ann = xp.asarray(ann_factor, dtype=mean.dtype)
    annualized_return = mean * ann
    # MDD == 0 → undefined Calmar. NaN in that case.
    abs_mdd = xp.abs(mdd)
    zero_dd_windows = int((abs_mdd == 0).sum())
    if zero_dd_windows > 0:
        warn_kuant(
            kernel="rollcalmar",
            code="KW-NUMERIC-ZERO-DRAWDOWN",
            what=(
                f"{zero_dd_windows} windows had zero drawdown; Calmar is "
                f"undefined there and set to NaN"
            ),
            fix=(
                "either accept NaN in monotonically-up regimes, or widen "
                "the window so a drawdown is captured"
            ),
            category=KuantNumericWarning,
        )
    safe_mdd = xp.where(abs_mdd > 0, abs_mdd, xp.asarray(xp.nan, dtype=mean.dtype))
    return annualized_return / safe_mdd

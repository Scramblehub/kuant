"""Hurst exponent via rescaled-range (R/S) analysis.

R/S analysis measures the self-similarity of a time series by
regressing the log of the mean rescaled range against the log of
the window size:

    z(t) = cumsum(r(t) - mean(r))
    R(w) = max(z within window w) - min(z within window w)
    S(w) = std(r within window w)
    mean_over_series log(R/S)(w) ~ H · log(w)

The slope H is the Hurst exponent. Range and interpretation:

    H = 0.5   random-walk / martingale behavior
    H > 0.5   persistent / trending (larger moves cluster together)
    H < 0.5   antipersistent / mean-reverting

Originally derived by H. E. Hurst (1951) for reservoir capacity
sizing on the Nile; brought into financial time-series analysis by
Mandelbrot & Wallis in the late 1960s.

Composes `rollrange` on the detrended cumulative series and
`rollstd` on the raw series.

Design: docs/kernels/stats/hurstrs.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError

from .rollrange import rollrange
from .rollstd import rollstd

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


@dataclass
class HurstResult:
    H: float
    windows: np.ndarray  # 1D int, the window sizes actually used
    log_rs: np.ndarray  # 1D float, mean log(R/S) at each window
    intercept: float
    n_windows: int

    def summary(self) -> str:
        return (
            f"Hurst exponent (R/S): H = {self.H:.4f}\n"
            f"  windows tested:     {self.n_windows} in [{int(self.windows[0])}, {int(self.windows[-1])}]\n"
            f"  intercept:          {self.intercept:+.4f}"
        )


def _to_numpy(r):
    if isinstance(r, _CUPY_NDARRAY):
        return r.get()
    return np.asarray(r, dtype=np.float64)


def hurstrs(r, min_w: int = 10, max_w: int | None = None, n_windows: int = 20) -> HurstResult:
    """Hurst exponent via rescaled-range analysis.

    Parameters
    ----------
    r : 1D array
        Returns (or any stationary series). NaN-tolerant.
    min_w : int, default 10
        Smallest window size to include in the log-log regression.
    max_w : int or None, default None
        Largest window size. Defaults to `len(r) // 4`.
    n_windows : int, default 20
        Approximate number of log-spaced window sizes between
        `min_w` and `max_w`. Duplicates from integer casting are
        deduplicated; the true count is in `result.n_windows`.

    Returns
    -------
    HurstResult
        With `H`, `windows`, `log_rs`, `intercept`, `n_windows`.

    Notes
    -----
    NaNs in the input propagate through `rollrange` and `rollstd`
    but are dropped when averaging log(R/S) at each window. Windows
    where every R/S sample is non-finite or non-positive contribute
    NaN to the regression and are excluded from the fit.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(2000)          # Brownian noise
    >>> result = hurstrs(r)
    >>> 0.35 < result.H < 0.65                 # near 0.5 for a random walk
    True
    """
    r_np = _to_numpy(r)
    require_1d(r_np, "r", kernel="hurstrs")
    n = r_np.size
    if n < 4 * min_w:
        raise KuantValueError(
            f"kuant.hurstrs: series too short for R/S with min_w={min_w}: "
            f"need >= {4 * min_w} samples, got {n}.  [KE-VAL-RANGE]\n"
            f"  → Fix: provide at least {4 * min_w} observations, or lower "
            f"`min_w` (default 10)"
        )

    if max_w is None:
        max_w = n // 4
    if max_w <= min_w:
        raise KuantValueError(
            f"kuant.hurstrs: 'max_w' ({max_w}) must exceed 'min_w' "
            f"({min_w}).  [KE-VAL-RANGE]\n"
            f"  → Fix: either raise `max_w` or lower `min_w`; the R/S fit "
            f"needs a range of window sizes to regress against"
        )

    ws = np.unique(np.logspace(np.log10(min_w), np.log10(max_w), n_windows).astype(int))
    if ws.size < 3:
        raise KuantValueError(
            f"kuant.hurstrs: too few distinct windows ({ws.size}); need "
            f">= 3 for the log-log regression.  [KE-VAL-RANGE]\n"
            f"  → Fix: increase `n_windows` (currently {n_windows}) or "
            f"widen the [min_w, max_w] range"
        )

    mean_r = float(np.nanmean(r_np))
    z = np.nancumsum(r_np - mean_r)

    log_rs = np.full(ws.size, np.nan)
    for i, w in enumerate(ws):
        R = rollrange(z, int(w))
        S = rollstd(r_np, int(w), ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = R / S
        rs = rs[np.isfinite(rs) & (rs > 0)]
        if rs.size:
            log_rs[i] = float(np.mean(np.log(rs)))

    log_ws = np.log(ws.astype(np.float64))
    valid = np.isfinite(log_rs)
    if valid.sum() < 3:
        raise KuantValueError(
            "kuant.hurstrs: fewer than 3 windows produced finite log(R/S) "
            "values, cannot fit the exponent.  [KE-CONV-DEGENERATE]\n"
            "  → Fix: input may be constant or near-constant on many "
            "windows (zero std). Check the series for long flat spans"
        )

    slope, intercept = np.polyfit(log_ws[valid], log_rs[valid], 1)
    return HurstResult(
        H=float(slope),
        windows=ws,
        log_rs=log_rs,
        intercept=float(intercept),
        n_windows=int(ws.size),
    )

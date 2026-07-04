"""Detrended Fluctuation Analysis (DFA) — long-memory exponent.

DFA (Peng et al. 1994) estimates the same self-similarity exponent H as
`hurstrs` but is more robust to nonstationarity in the mean. The
algorithm:

  1. Cumulative-sum the mean-centered series: Y(k) = Σ (x_i - mean).
  2. For each window size w:
     - split Y into non-overlapping windows of length w
     - in each window, fit a linear detrend Y ~ a + b·t
     - RMS residual after detrend = F(w)
  3. Regress log(F(w)) on log(w). Slope = H.

    H ~ 0.5   random walk
    H > 0.5   persistent / trending
    H < 0.5   antipersistent / mean-reverting

Design: docs/kernels/stats/dfa.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError


@dataclass
class DFAResult:
    H: float
    windows: np.ndarray
    log_F: np.ndarray
    intercept: float
    n_windows: int

    def summary(self) -> str:
        return (
            f"DFA exponent (Peng): H = {self.H:.4f}\n"
            f"  windows tested:   {self.n_windows} in [{int(self.windows[0])}, "
            f"{int(self.windows[-1])}]\n"
            f"  intercept:        {self.intercept:+.4f}"
        )


def dfa(x, min_w: int = 10, max_w: int | None = None, n_windows: int = 20) -> DFAResult:
    """Detrended Fluctuation Analysis on a 1D series.

    Parameters
    ----------
    x : 1D array
        Series to analyze. NaN-tolerant (dropped from the mean).
    min_w : int, default 10
        Smallest window size.
    max_w : int or None, default None
        Largest window size. Defaults to `len(x) // 4`.
    n_windows : int, default 20
        Approximate log-spaced window count.

    Returns
    -------
    DFAResult with `H`, `windows`, `log_F`, `intercept`, `n_windows`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(2000)   # random walk in cumsum
    >>> result = dfa(r)
    >>> 0.35 < result.H < 0.65
    True
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="dfa")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 4 * min_w:
        raise KuantValueError(
            f"kuant.dfa: series too short for DFA with min_w={min_w}: need "
            f">= {4 * min_w} clean samples, got {n} after NaN drop.  "
            f"[KE-VAL-RANGE]\n"
            f"  → Fix: provide at least {4 * min_w} finite observations, "
            f"or lower `min_w` (default 4)"
        )
    if max_w is None:
        max_w = n // 4
    if max_w <= min_w:
        raise KuantValueError(
            f"kuant.dfa: 'max_w' ({max_w}) must exceed 'min_w' ({min_w}).  "
            f"[KE-VAL-RANGE]\n"
            f"  → Fix: either raise `max_w` or lower `min_w`; the DFA fit "
            f"needs a range of window sizes to regress against"
        )

    # 1) Integrate mean-centered series
    y = np.cumsum(arr - float(np.mean(arr)))

    # 2) Log-spaced windows
    ws = np.unique(np.logspace(np.log10(min_w), np.log10(max_w), n_windows).astype(int))
    if ws.size < 3:
        raise KuantValueError(
            f"kuant.dfa: too few distinct windows ({ws.size}); need >= 3 "
            f"for the log-log regression.  [KE-VAL-RANGE]\n"
            f"  → Fix: increase `n_windows` (currently {n_windows}) or "
            f"widen the [min_w, max_w] range"
        )

    log_F = np.full(ws.size, np.nan)
    for i, w in enumerate(ws):
        n_windows_here = n // w
        if n_windows_here < 1:
            continue
        # Slice into shape (n_windows_here, w)
        y_sliced = y[: n_windows_here * w].reshape(n_windows_here, w)
        # Linear detrend within each window (fit y ~ a + b·t)
        t = np.arange(w, dtype=np.float64)
        # Vectorized OLS: slope = cov(t, y) / var(t)
        t_mean = t.mean()
        y_mean = y_sliced.mean(axis=1, keepdims=True)
        t_c = t - t_mean
        y_c = y_sliced - y_mean
        slope = (y_c * t_c).sum(axis=1) / (t_c * t_c).sum()
        intercept = y_mean.squeeze(-1) - slope * t_mean
        trend = intercept[:, None] + slope[:, None] * t[None, :]
        residual = y_sliced - trend
        # RMS across time within each window, then average across windows
        rms = np.sqrt((residual * residual).mean(axis=1))
        F_w = float(rms.mean())
        if F_w > 0:
            log_F[i] = np.log(F_w)

    log_ws = np.log(ws.astype(np.float64))
    valid = np.isfinite(log_F)
    if valid.sum() < 3:
        raise KuantValueError(
            "kuant.dfa: fewer than 3 windows produced finite F(w) values, "
            "cannot fit the exponent.  [KE-CONV-DEGENERATE]\n"
            "  → Fix: input may be constant on many windows (zero RMS); "
            "check the series for long flat spans"
        )
    slope, intercept = np.polyfit(log_ws[valid], log_F[valid], 1)
    return DFAResult(
        H=float(slope),
        windows=ws,
        log_F=log_F,
        intercept=float(intercept),
        n_windows=int(ws.size),
    )

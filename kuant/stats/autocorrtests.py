"""Autocorrelation portmanteau + first-order tests.

Three staples of time-series residual diagnostics:

- `ljungbox` (Ljung-Box 1978): most-used portmanteau test for absence
  of autocorrelation up to lag `h`.
- `boxpierce` (Box-Pierce 1970): the original portmanteau. Less
  small-sample power than Ljung-Box but still widely reported for
  legacy comparison.
- `durbinwatson`: first-order autocorrelation test. Fast smoke check
  on regression residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class PortmanteauResult:
    stat: float
    p_value: float
    h: int
    dof: int
    test: str

    def summary(self) -> str:
        return (
            f"=== PortmanteauResult ({self.test}) ===\n"
            f"stat:       {self.stat:.4f}\n"
            f"p-value:    {self.p_value:.4g}\n"
            f"h / dof:    {self.h} / {self.dof}"
        )


@dataclass
class DurbinWatsonResult:
    stat: float
    n: int

    def summary(self) -> str:
        interp = "no first-order autocorrelation"
        if self.stat < 1.5:
            interp = "positive first-order autocorrelation"
        elif self.stat > 2.5:
            interp = "negative first-order autocorrelation"
        return (
            "=== DurbinWatsonResult ===\n"
            f"stat:  {self.stat:.4f}  ({interp})\n"
            f"n:     {self.n}"
        )


def _acf(x, max_lag):
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom < 1e-15:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        acf[k] = np.sum(x[k:] * x[:-k]) / denom
    return acf


def _chi2_sf(x: float, df: int) -> float:
    # Survival function 1 - CDF via scipy if available; else Wilson-Hilferty.
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, df))
    except ImportError:
        # Wilson-Hilferty transform: (x/df)^(1/3) ~ N(1 - 2/(9df), 2/(9df))
        # then normal-CDF via erf.
        from math import erf, sqrt

        h = 2.0 / (9.0 * df)
        z = ((x / df) ** (1.0 / 3.0) - (1.0 - h)) / sqrt(h)
        # 1 - Phi(z) via erf.
        return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def ljungbox(x, *, h: int = 10, dof_correction: int = 0) -> PortmanteauResult:
    """Ljung-Box portmanteau test.

    Parameters
    ----------
    x : 1D array (residuals or a stationary series)
    h : int, default 10
        Number of autocorrelation lags to include.
    dof_correction : int, default 0
        Subtract the number of estimated ARMA parameters when testing
        on ARMA residuals; degrees-of-freedom becomes `h - dof_correction`.

    Returns
    -------
    PortmanteauResult

    References
    ----------
    Ljung & Box 1978, "On a measure of lack of fit in time series models."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="ljungbox")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 20:
        raise KuantValueError(
            f"kuant.ljungbox: only {n} finite values; need at least 20.  " f"[KE-VAL-MIN-CLEAN]"
        )
    require_range(h, "h", kernel="ljungbox", lo=1, hi=n - 1)
    if dof_correction >= h:
        raise KuantValueError(
            f"kuant.ljungbox: dof_correction ({dof_correction}) must be < h "
            f"({h}).  [KE-VAL-RANGE]"
        )
    acf = _acf(arr, int(h))
    # LB statistic.
    lags = np.arange(1, int(h) + 1)
    stat = n * (n + 2) * np.sum(acf[1:] ** 2 / (n - lags))
    dof = int(h) - int(dof_correction)
    p_value = _chi2_sf(float(stat), dof)
    return PortmanteauResult(
        stat=float(stat),
        p_value=float(p_value),
        h=int(h),
        dof=dof,
        test="ljung-box",
    )


def boxpierce(x, *, h: int = 10, dof_correction: int = 0) -> PortmanteauResult:
    """Box-Pierce portmanteau test."""
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="boxpierce")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 20:
        raise KuantValueError(
            f"kuant.boxpierce: only {n} finite values; need at least 20.  " f"[KE-VAL-MIN-CLEAN]"
        )
    require_range(h, "h", kernel="boxpierce", lo=1, hi=n - 1)
    if dof_correction >= h:
        raise KuantValueError(
            f"kuant.boxpierce: dof_correction ({dof_correction}) must be < h "
            f"({h}).  [KE-VAL-RANGE]"
        )
    acf = _acf(arr, int(h))
    stat = n * np.sum(acf[1:] ** 2)
    dof = int(h) - int(dof_correction)
    p_value = _chi2_sf(float(stat), dof)
    return PortmanteauResult(
        stat=float(stat),
        p_value=float(p_value),
        h=int(h),
        dof=dof,
        test="box-pierce",
    )


def durbinwatson(x) -> DurbinWatsonResult:
    """Durbin-Watson first-order autocorrelation statistic.

    DW = sum((e_t - e_{t-1})^2) / sum(e_t^2).
    Values near 2 indicate no autocorrelation; near 0 = positive; near 4 = negative.
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="durbinwatson")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 20:
        raise KuantValueError(
            f"kuant.durbinwatson: only {n} finite values; need at least 20.  " f"[KE-VAL-MIN-CLEAN]"
        )
    diff = np.diff(arr)
    denom = float(np.sum(arr * arr))
    if denom < 1e-15:
        return DurbinWatsonResult(stat=float("nan"), n=int(n))
    stat = float(np.sum(diff * diff) / denom)
    return DurbinWatsonResult(stat=stat, n=int(n))


__all__ = ["PortmanteauResult", "DurbinWatsonResult", "ljungbox", "boxpierce", "durbinwatson"]

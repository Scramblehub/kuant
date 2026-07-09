"""Normality tests for time series and residuals.

Three widely-cited tests:

- `jarquebera`: skewness + excess-kurtosis based, chi-square-2 null.
  Fast; the standard reported alongside GARCH / OLS output.
- `andersondarling`: EDF-based, weighted to tails. More powerful in
  the tails than Kolmogorov-Smirnov.
- `shapirowilk`: sample-order-statistic based. Most powerful for small
  samples but capped at n <= 5000 in most implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError


@dataclass
class NormalityResult:
    stat: float
    p_value: float
    n: int
    test: str
    extra: dict

    def summary(self) -> str:
        extra_str = "  ".join(f"{k}={v:.4g}" for k, v in self.extra.items())
        return (
            f"=== NormalityResult ({self.test}) ===\n"
            f"stat:      {self.stat:.4f}\n"
            f"p-value:   {self.p_value:.4g}\n"
            f"n:         {self.n}\n"
            f"{extra_str}"
        )


def _chi2_sf(x: float, df: int) -> float:
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, df))
    except ImportError:
        from math import erf, sqrt

        h = 2.0 / (9.0 * df)
        z = ((x / df) ** (1.0 / 3.0) - (1.0 - h)) / sqrt(h)
        return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def jarquebera(x) -> NormalityResult:
    """Jarque-Bera normality test.

    JB = n/6 * (S^2 + (K - 3)^2 / 4), null hypothesis Gaussian,
    distributed as chi-square with 2 degrees of freedom under H0.

    Parameters
    ----------
    x : 1D array

    Returns
    -------
    NormalityResult

    References
    ----------
    Jarque & Bera 1980.
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="jarquebera")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 20:
        raise KuantValueError(
            f"kuant.jarquebera: only {n} finite values; need at least 20.  " f"[KE-VAL-MIN-CLEAN]"
        )
    mu = arr.mean()
    sd = arr.std(ddof=0)
    if sd < 1e-12:
        return NormalityResult(
            stat=float("nan"),
            p_value=float("nan"),
            n=int(n),
            test="jarque-bera",
            extra={"skew": float("nan"), "kurt": float("nan")},
        )
    z = (arr - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    jb = n / 6.0 * (skew**2 + (kurt - 3.0) ** 2 / 4.0)
    p = _chi2_sf(float(jb), 2)
    return NormalityResult(
        stat=float(jb),
        p_value=float(p),
        n=int(n),
        test="jarque-bera",
        extra={"skew": skew, "kurt": kurt},
    )


def andersondarling(x) -> NormalityResult:
    """Anderson-Darling normality test.

    Returns the A2 statistic (with the standard small-sample correction)
    and a p-value approximation from Stephens 1986.

    Parameters
    ----------
    x : 1D array

    Returns
    -------
    NormalityResult

    References
    ----------
    Anderson & Darling 1954; Stephens 1986.
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="andersondarling")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 20:
        raise KuantValueError(
            f"kuant.andersondarling: only {n} finite values; need at least "
            f"20.  [KE-VAL-MIN-CLEAN]"
        )
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd < 1e-12:
        return NormalityResult(
            stat=float("nan"),
            p_value=float("nan"),
            n=int(n),
            test="anderson-darling",
            extra={},
        )
    z = np.sort((arr - mu) / sd)

    try:
        from scipy.stats import norm

        F = norm.cdf(z)
    except ImportError:
        from math import erf, sqrt as msqrt

        F = 0.5 * (1.0 + np.array([erf(v / msqrt(2)) for v in z]))

    F = np.clip(F, 1e-12, 1 - 1e-12)
    idx = np.arange(1, n + 1)
    A2 = -n - np.sum((2 * idx - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n
    # Small-sample correction.
    A2_adj = A2 * (1 + 0.75 / n + 2.25 / (n * n))
    # Approximate p-value (Stephens 1986).
    if A2_adj < 0.2:
        p = 1 - np.exp(-13.436 + 101.14 * A2_adj - 223.73 * A2_adj**2)
    elif A2_adj < 0.34:
        p = 1 - np.exp(-8.318 + 42.796 * A2_adj - 59.938 * A2_adj**2)
    elif A2_adj < 0.6:
        p = np.exp(0.9177 - 4.279 * A2_adj - 1.38 * A2_adj**2)
    else:
        p = np.exp(1.2937 - 5.709 * A2_adj + 0.0186 * A2_adj**2)
    return NormalityResult(
        stat=float(A2_adj),
        p_value=float(p),
        n=int(n),
        test="anderson-darling",
        extra={},
    )


def shapirowilk(x) -> NormalityResult:
    """Shapiro-Wilk normality test.

    Thin wrapper around `scipy.stats.shapiro` (which caps at n <= 5000).
    Raises `KuantValueError` if scipy is not installed.

    Parameters
    ----------
    x : 1D array

    Returns
    -------
    NormalityResult
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="shapirowilk")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 20:
        raise KuantValueError(
            f"kuant.shapirowilk: only {n} finite values; need at least 20.  " f"[KE-VAL-MIN-CLEAN]"
        )
    if n > 5000:
        # Standard scipy cap; downsample deterministically.
        idx = np.linspace(0, n - 1, 5000).astype(int)
        arr = arr[idx]
    try:
        from scipy.stats import shapiro
    except ImportError as e:
        raise KuantValueError(
            "kuant.shapirowilk: requires scipy.  [KE-DEP-MISSING]\n" "  -> Fix: pip install scipy"
        ) from e
    stat, p = shapiro(arr)
    return NormalityResult(
        stat=float(stat),
        p_value=float(p),
        n=int(arr.size),
        test="shapiro-wilk",
        extra={},
    )


__all__ = ["NormalityResult", "jarquebera", "andersondarling", "shapirowilk"]

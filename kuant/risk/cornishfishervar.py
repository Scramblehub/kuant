"""Cornish-Fisher expansion VaR.

Adjusts the Gaussian VaR quantile for empirical skewness and excess
kurtosis via the Cornish-Fisher 1937 series. When returns have fat
tails or non-zero skew, this yields a materially different VaR than
the naive Gaussian formula.

Formula:
    z_cf = z + (z^2 - 1) * S / 6
             + (z^3 - 3z) * K / 24
             - (2 z^3 - 5z) * S^2 / 36
    VaR_alpha = -(mu + z_cf * sigma)

where S is sample skew and K is sample excess kurtosis (fourth central
moment / sigma^4 - 3), z is the Gaussian quantile at level (1 - alpha).

Note the sign convention: VaR is reported as a POSITIVE loss magnitude.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class CornishFisherVarResult:
    var: float
    z_gaussian: float
    z_cf: float
    mean: float
    std: float
    skew: float
    excess_kurtosis: float
    alpha: float
    n: int

    def summary(self) -> str:
        return (
            "=== CornishFisherVarResult ===\n"
            f"VaR (loss):          {self.var:+.6f}\n"
            f"alpha:               {self.alpha}\n"
            f"z (gaussian / CF):   {self.z_gaussian:+.4f} / {self.z_cf:+.4f}\n"
            f"mean / std:          {self.mean:+.6f} / {self.std:.6f}\n"
            f"skew / excess kurt:  {self.skew:+.4f} / {self.excess_kurtosis:+.4f}\n"
            f"n:                   {self.n}"
        )


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via scipy if available, else a rational approx."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except ImportError:
        # Beasley-Springer-Moro rational approximation
        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
        p_low, p_high = 0.02425, 1 - 0.02425
        if p < p_low:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
            )
        if p <= p_high:
            q = p - 0.5
            r = q * q
            return (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
            )
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )


def cornishfishervar(returns, *, alpha: float = 0.95) -> CornishFisherVarResult:
    """Cornish-Fisher expansion VaR.

    Parameters
    ----------
    returns : 1D array
    alpha : float, default 0.95
        Confidence level (typical: 0.95 or 0.99).

    Returns
    -------
    CornishFisherVarResult

    References
    ----------
    Cornish & Fisher 1937, "Moments and cumulants in the specification
    of distributions."
    """
    arr = np.asarray(returns, dtype=np.float64)
    require_1d(arr, "returns", kernel="cornishfishervar")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 30:
        raise KuantValueError(
            f"kuant.cornishfishervar: only {n} finite values; need at "
            f"least 30 for a stable Cornish-Fisher estimate.  "
            f"[KE-VAL-MIN-CLEAN]"
        )
    require_range(alpha, "alpha", kernel="cornishfishervar", lo=0.5, hi=0.999)

    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    if sigma < 1e-15:
        # Degenerate constant series: no dispersion. VaR is conventionally
        # a non-negative loss magnitude, so clamp -mu at 0 (mu > 0 would
        # otherwise report a "negative loss" here).
        return CornishFisherVarResult(
            var=max(-mu, 0.0),
            z_gaussian=float("nan"),
            z_cf=float("nan"),
            mean=mu,
            std=sigma,
            skew=float("nan"),
            excess_kurtosis=float("nan"),
            alpha=float(alpha),
            n=int(n),
        )
    z = (arr - mu) / sigma
    S = float(np.mean(z**3))
    K = float(np.mean(z**4) - 3.0)

    if abs(S) > 1.0 or abs(K) > 7.0:
        warnings.warn(
            f"kuant.cornishfishervar: sample skew ({S:+.2f}) or excess "
            f"kurtosis ({K:+.2f}) is outside the Cornish-Fisher expansion's "
            f"safe range (|skew|<=1, |excess kurt|<=7). The returned VaR "
            f"is unreliable and may be non-monotone in alpha. Prefer "
            f"kuant.risk.evtvar for heavy-tailed series.  "
            f"[KW-CF-EXPANSION-INVALID]",
            KuantNumericWarning,
            stacklevel=2,
        )
    z_gauss = _norm_ppf(1.0 - alpha)  # negative for standard alpha ~ 0.95
    z_cf = (
        z_gauss
        + (z_gauss**2 - 1) * S / 6
        + (z_gauss**3 - 3 * z_gauss) * K / 24
        - (2 * z_gauss**3 - 5 * z_gauss) * S**2 / 36
    )
    var = -(mu + z_cf * sigma)
    return CornishFisherVarResult(
        var=float(var),
        z_gaussian=float(z_gauss),
        z_cf=float(z_cf),
        mean=mu,
        std=sigma,
        skew=S,
        excess_kurtosis=K,
        alpha=float(alpha),
        n=int(n),
    )


__all__ = ["CornishFisherVarResult", "cornishfishervar"]

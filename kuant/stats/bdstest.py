"""BDS test for iid / nonlinear structure detection.

The Brock-Dechert-Scheinkman (1996) test uses the correlation integral
across embedding dimensions to detect nonlinear dependence when linear
autocorrelation is absent. A powerful "residual after linear filtering
is still not iid" test.

Under H0 (x is iid), the BDS statistic is asymptotically N(0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class BdsResult:
    stat: float
    p_value: float
    m: int
    epsilon: float
    n: int

    def summary(self) -> str:
        return (
            "=== BdsResult ===\n"
            f"stat (z):   {self.stat:+.4f}\n"
            f"p-value:    {self.p_value:.4g}\n"
            f"m:          {self.m}\n"
            f"epsilon:    {self.epsilon:.4g}\n"
            f"n:          {self.n}"
        )


def _norm_sf(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def _correlation_integral(x, m, eps):
    """Standard BDS correlation integral C_m(eps) via pairwise Chebyshev
    distance count."""
    n = x.size - m + 1
    if n < 2:
        return 0.0
    # Embed.
    emb = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        emb[:, j] = x[j : j + n]
    # Count pairs i < j with sup-norm dist <= eps.
    count = 0
    for i in range(n):
        diff = np.max(np.abs(emb[i + 1 :] - emb[i]), axis=1)
        count += int(np.sum(diff <= eps))
    total = n * (n - 1) / 2.0
    return count / total if total > 0 else 0.0


def bdstest(x, *, m: int = 2, epsilon: float | None = None) -> BdsResult:
    """BDS iid / nonlinearity test.

    Parameters
    ----------
    x : 1D array
    m : int, default 2
        Embedding dimension.
    epsilon : float, optional
        Distance threshold. Default `0.7 * std(x)` per Kanzler 1999.

    Returns
    -------
    BdsResult

    References
    ----------
    Brock, Dechert, Scheinkman & LeBaron 1996, "A test for independence
    based on the correlation dimension."

    Notes
    -----
    The variance estimator uses a simplified form of the full Brock et al
    1996 formula that omits higher-order moment corrections. For m >= 3
    the reported z-statistic is conservative (understates significance).
    m=2 recovers the exact form. For rigorous inference at higher m,
    combine with a bootstrap null via `kuant.nulltest`.
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="bdstest")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 100:
        raise KuantValueError(
            f"kuant.bdstest: only {n} finite values; need at least 100.  " f"[KE-VAL-MIN-CLEAN]"
        )
    require_range(m, "m", kernel="bdstest", lo=2, hi=10)
    if epsilon is None:
        epsilon = 0.7 * float(np.std(arr, ddof=1))
    if epsilon <= 0:
        raise KuantValueError(
            f"kuant.bdstest: 'epsilon' must be positive, got {epsilon}.  " f"[KE-VAL-POSITIVE]"
        )

    C_m = _correlation_integral(arr, int(m), float(epsilon))
    C_1 = _correlation_integral(arr, 1, float(epsilon))
    if C_1 <= 0 or C_m <= 0:
        return BdsResult(
            stat=float("nan"),
            p_value=float("nan"),
            m=int(m),
            epsilon=float(epsilon),
            n=int(n),
        )
    # Standard BDS asymptotic variance (Brock et al 1996 eqn 2.5).
    # Simplified numerator: sqrt(n) * (C_m - C_1^m).
    stat_num = np.sqrt(n) * (C_m - C_1**m)
    # Variance approximation using K = C_1(2*eps).
    K = _correlation_integral(arr, 1, 2 * float(epsilon))
    var = 4 * (
        K**m
        + 2 * sum(K ** (m - j) * C_1 ** (2 * j) for j in range(1, m))
        + (m - 1) ** 2 * C_1 ** (2 * m)
        - m**2 * K * C_1 ** (2 * m - 2)
    )
    if var <= 0:
        return BdsResult(
            stat=float("nan"),
            p_value=float("nan"),
            m=int(m),
            epsilon=float(epsilon),
            n=int(n),
        )
    stat = float(stat_num / np.sqrt(var))
    p = 2.0 * _norm_sf(abs(stat))
    return BdsResult(
        stat=stat,
        p_value=float(p),
        m=int(m),
        epsilon=float(epsilon),
        n=int(n),
    )


__all__ = ["BdsResult", "bdstest"]

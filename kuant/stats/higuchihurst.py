"""Higuchi 1988 fractal-dimension method for Hurst exponent.

Estimates the fractal dimension D of a time series via the k-step
curve length; then reports Hurst as H = 2 - D. Compared to R/S:

- More stable on short series (~200-500 points).
- Less sensitive to non-stationarity.
- Faster: O(N * k_max) instead of R/S's O(N * n_windows).

Best combined with `hurstrs` and `dfa` as a cross-check; disagreement
across estimators is a real signal about the series' regularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class HiguchiHurstResult:
    hurst: float
    fractal_dim: float
    log_k: np.ndarray
    log_L: np.ndarray
    intercept: float
    k_max: int

    def summary(self) -> str:
        return (
            "=== HiguchiHurstResult ===\n"
            f"Hurst H:         {self.hurst:.4f}\n"
            f"fractal dim D:   {self.fractal_dim:.4f}\n"
            f"k_max:           {self.k_max}\n"
            f"log-log points:  {self.log_k.size}"
        )


_DEFAULT_KMAX = 30


def higuchihurst(x, *, k_max: int = _DEFAULT_KMAX) -> HiguchiHurstResult:
    """Higuchi fractal-dimension estimator for the Hurst exponent.

    Parameters
    ----------
    x : 1D array
    k_max : int, default 30
        Maximum step size in the curve-length calculation.

    Returns
    -------
    HiguchiHurstResult

    References
    ----------
    Higuchi 1988, "Approach to an irregular time series on the basis of
    the fractal theory."

    Notes
    -----
    Convention: Higuchi's method returns fractal dimension D in [1, 2]
    and reports Hurst as H = 2 - D. For pure white noise, D approaches 2
    so H approaches 0 (not 0.5, unlike R/S). To compare cross-method,
    use `hurstrs` (H ~ 0.5 for noise) or `wavelethurst` on the SAME
    input to see the different self-similarity notions.
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="higuchihurst")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 100:
        raise KuantValueError(
            f"kuant.higuchihurst: only {n} finite values; need at least "
            f"100.  [KE-VAL-MIN-CLEAN]"
        )
    # When the caller relied on the default k_max=30, cap it to n//4 so the
    # default stays feasible in the n=100..119 range. Explicit user-supplied
    # values still go through require_range and can raise KE-VAL-RANGE.
    if int(k_max) == _DEFAULT_KMAX:
        k_max = min(int(k_max), n // 4)
    require_range(k_max, "k_max", kernel="higuchihurst", lo=4, hi=n // 4)

    L_k = np.zeros(int(k_max), dtype=np.float64)
    for k in range(1, int(k_max) + 1):
        # Average curve length over all m in [1, k].
        L_m_vals = []
        for m in range(1, k + 1):
            # Indices: m, m+k, m+2k, ...
            n_steps = (n - m) // k
            if n_steps < 1:
                continue
            idx = m - 1 + k * np.arange(n_steps + 1)
            if idx[-1] >= n:
                idx = idx[:-1]
            if idx.size < 2:
                continue
            sub = arr[idx]
            L_m = np.sum(np.abs(np.diff(sub))) * (n - 1) / ((idx.size - 1) * k * k)
            L_m_vals.append(L_m)
        if L_m_vals:
            L_k[k - 1] = float(np.mean(L_m_vals))
        else:
            L_k[k - 1] = np.nan

    ks = np.arange(1, int(k_max) + 1, dtype=np.float64)
    valid = np.isfinite(L_k) & (L_k > 0)
    if valid.sum() < 4:
        raise KuantValueError(
            "kuant.higuchihurst: insufficient valid log-log points; "
            "increase 'k_max' or provide more data.  [KE-VAL-MIN-CLEAN]"
        )
    log_k = np.log(ks[valid])
    log_L = np.log(L_k[valid])
    slope, intercept = np.polyfit(log_k, log_L, 1)
    D = -float(slope)
    H = 2.0 - D
    return HiguchiHurstResult(
        hurst=float(H),
        fractal_dim=float(D),
        log_k=log_k,
        log_L=log_L,
        intercept=float(intercept),
        k_max=int(k_max),
    )


__all__ = ["HiguchiHurstResult", "higuchihurst"]

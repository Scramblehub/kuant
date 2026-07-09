"""Abry-Veitch wavelet-based Hurst exponent.

For a self-similar process with Hurst H, the variance of the discrete
wavelet coefficients at scale j scales as `var_j ~ 2 ** (j * (2H - 1))`.
Fitting `log2(var_j) = j * (2H - 1) + const` gives H from the slope.

Implementation uses a Haar wavelet decomposition to keep the file
self-contained (no PyWavelets dependency). The Haar wavelet is
sufficient for Hurst estimation; other wavelets improve the constant
factor but not the asymptotic scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class WaveletHurstResult:
    hurst: float
    scales: np.ndarray
    log_var: np.ndarray
    intercept: float

    def summary(self) -> str:
        return (
            "=== WaveletHurstResult ===\n"
            f"Hurst H:       {self.hurst:.4f}\n"
            f"scales used:   {list(self.scales)}\n"
            f"intercept:     {self.intercept:.4f}"
        )


def _haar_dwt_coefs_by_scale(x: np.ndarray, n_scales: int):
    """Non-redundant Haar DWT: returns detail coefs per scale."""
    coefs = []
    cur = x.copy()
    for _ in range(n_scales):
        if cur.size < 2:
            break
        # Truncate to even length.
        if cur.size % 2 == 1:
            cur = cur[:-1]
        c1 = cur[::2]
        c2 = cur[1::2]
        # Haar detail (difference) and approximation (sum).
        detail = (c1 - c2) / np.sqrt(2)
        approx = (c1 + c2) / np.sqrt(2)
        coefs.append(detail)
        cur = approx
    return coefs


def wavelethurst(x, *, scale_lo: int = 2, scale_hi: int = 7) -> WaveletHurstResult:
    """Abry-Veitch wavelet Hurst estimator (Haar wavelet).

    Parameters
    ----------
    x : 1D array
    scale_lo, scale_hi : int
        Range of dyadic scales to include in the log-log fit. Default
        2 to 7 covers 4 to 128 sample resolutions.

    Returns
    -------
    WaveletHurstResult

    References
    ----------
    Abry & Veitch 1998, "Wavelet analysis of long-range-dependent
    traffic."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="wavelethurst")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 128:
        raise KuantValueError(
            f"kuant.wavelethurst: only {n} finite values; need at least "
            f"128 for a stable dyadic decomposition.  [KE-VAL-MIN-CLEAN]"
        )
    require_range(scale_lo, "scale_lo", kernel="wavelethurst", lo=1, hi=15)
    require_range(scale_hi, "scale_hi", kernel="wavelethurst", lo=scale_lo + 1, hi=20)
    max_possible_scales = int(np.floor(np.log2(n)))
    n_scales = min(int(scale_hi), max_possible_scales)
    if n_scales <= int(scale_lo):
        raise KuantValueError(
            f"kuant.wavelethurst: with n={n}, max scale is {max_possible_scales}; "
            f"scale_lo ({scale_lo}) leaves no fit range.  [KE-VAL-RANGE]"
        )
    coefs = _haar_dwt_coefs_by_scale(arr, n_scales)
    # Variance at each scale.
    vars_by_scale = np.array([np.var(c, ddof=1) if c.size > 1 else np.nan for c in coefs])
    scales = np.arange(1, len(coefs) + 1)
    # Restrict to fit range.
    fit_mask = (scales >= int(scale_lo)) & (scales <= n_scales)
    fit_mask &= np.isfinite(vars_by_scale) & (vars_by_scale > 0)
    if fit_mask.sum() < 3:
        raise KuantValueError(
            "kuant.wavelethurst: fewer than 3 valid scales in the fit " "range.  [KE-VAL-MIN-CLEAN]"
        )
    log_var = np.log2(vars_by_scale[fit_mask])
    fit_scales = scales[fit_mask].astype(np.float64)
    slope, intercept = np.polyfit(fit_scales, log_var, 1)
    # var_j ~ 2^(j*(2H - 1)) -> slope = 2H - 1 -> H = (slope + 1) / 2.
    H = (slope + 1.0) / 2.0
    return WaveletHurstResult(
        hurst=float(H),
        scales=fit_scales.astype(int),
        log_var=log_var,
        intercept=float(intercept),
    )


__all__ = ["WaveletHurstResult", "wavelethurst"]

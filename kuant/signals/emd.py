"""Empirical Mode Decomposition (Huang 1998).

Adaptive time-series decomposition into intrinsic mode functions
(IMFs) via iterative sifting. Unlike wavelets, EMD does not require
choosing a basis: the decomposition is data-driven.

Pipeline (per IMF):
1. Identify local maxima and minima.
2. Interpolate to form upper / lower envelopes (cubic spline).
3. Subtract the envelope mean from the signal.
4. Repeat until the residual satisfies IMF criteria (equal max count
   and zero crossings; envelope mean near zero).

This MVP uses a fixed-iteration sifting scheme; production-grade
ensembles (EEMD, CEEMDAN) are deferred to a later release.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive
from kuant.errors import KuantValueError


@dataclass
class EmdResult:
    imfs: list  # list of 1D arrays
    residual: np.ndarray
    n_imfs: int

    def summary(self) -> str:
        return (
            "=== EmdResult ===\n"
            f"n IMFs:     {self.n_imfs}\n"
            f"IMF vars:   {[float(np.var(imf, ddof=1)) for imf in self.imfs]}\n"
            f"residual std: {float(np.std(self.residual, ddof=1)):.4f}"
        )


def _find_extrema(x):
    d = np.diff(x)
    # Local max: d > 0 then d < 0 (idx i where d[i-1] > 0 > d[i])
    max_idx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    min_idx = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return max_idx, min_idx


def _sift_once(x):
    n = x.size
    max_idx, min_idx = _find_extrema(x)
    if max_idx.size < 2 or min_idx.size < 2:
        return None
    # Cubic spline envelopes over endpoints.
    try:
        from scipy.interpolate import CubicSpline
    except ImportError as e:
        raise KuantValueError(
            "kuant.emd: requires scipy.interpolate.  [KE-DEP-MISSING]\n"
            "  -> Fix: pip install scipy"
        ) from e

    # Add endpoints to stabilize the spline.
    max_x = np.concatenate([[0], max_idx, [n - 1]])
    max_y = np.concatenate([[x[0]], x[max_idx], [x[-1]]])
    min_x = np.concatenate([[0], min_idx, [n - 1]])
    min_y = np.concatenate([[x[0]], x[min_idx], [x[-1]]])
    upper = CubicSpline(max_x, max_y)(np.arange(n))
    lower = CubicSpline(min_x, min_y)(np.arange(n))
    mean_env = 0.5 * (upper + lower)
    return x - mean_env


def emd(x, *, max_imfs: int = 8, sifting_iters: int = 10) -> EmdResult:
    """Empirical mode decomposition into intrinsic mode functions.

    Parameters
    ----------
    x : 1D array
    max_imfs : int, default 8
        Cap on the number of IMFs to extract.
    sifting_iters : int, default 10
        Number of sifting iterations per IMF. Fixed-iteration variant.

    Returns
    -------
    EmdResult

    References
    ----------
    Huang et al 1998, "The empirical mode decomposition and the Hilbert
    spectrum for nonlinear and non-stationary time series analysis."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="emd")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 64:
        raise KuantValueError(
            f"kuant.emd: only {n} finite values; need at least 64 for a "
            f"stable decomposition.  [KE-VAL-MIN-CLEAN]"
        )
    require_positive(max_imfs, "max_imfs", kernel="emd", kind="int")
    require_positive(sifting_iters, "sifting_iters", kernel="emd", kind="int")

    residual = arr.copy()
    imfs = []
    for _ in range(int(max_imfs)):
        candidate = residual.copy()
        for _sift in range(int(sifting_iters)):
            sifted = _sift_once(candidate)
            if sifted is None:
                break
            candidate = sifted
        if _sift_once(candidate) is None:
            break
        imfs.append(candidate.copy())
        residual = residual - candidate
        # If residual is monotone (no extrema), stop.
        max_i, min_i = _find_extrema(residual)
        if max_i.size < 2 or min_i.size < 2:
            break

    return EmdResult(imfs=imfs, residual=residual, n_imfs=len(imfs))


__all__ = ["EmdResult", "emd"]

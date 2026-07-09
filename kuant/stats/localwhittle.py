"""Local Whittle estimator for long-memory parameter d (Hurst H = d + 0.5).

Fits a semiparametric long-memory model to the low-frequency portion
of the periodogram. The local Whittle likelihood is:

    L(d) = -log(mean(w_j ** (2d) * I_j)) + (2d / m) * sum(log w_j)

where I_j is the periodogram at Fourier frequency w_j and the sum is
taken over the first m frequencies. Minimizing L(d) numerically gives
the estimator.

Compared to R/S and DFA:
- Sharper asymptotic efficiency for pure long-memory processes
- Less sensitive to short-memory contamination
- Well-defined confidence intervals under Gaussian assumptions
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class LocalWhittleResult:
    d: float
    hurst: float
    m: int
    n: int
    se: float

    def summary(self) -> str:
        return (
            "=== LocalWhittleResult ===\n"
            f"d (long memory):   {self.d:+.4f}\n"
            f"Hurst H (d + 0.5): {self.hurst:.4f}\n"
            f"m (frequencies):   {self.m}\n"
            f"n:                 {self.n}\n"
            f"SE (asymptotic):   {self.se:.4f}"
        )


def localwhittle(x, *, m: int | None = None) -> LocalWhittleResult:
    """Robinson 1995 local Whittle long-memory estimator.

    Parameters
    ----------
    x : 1D array
    m : int, optional
        Number of Fourier frequencies to include. Default: `n ** 0.7`
        (Robinson recommended rate for optimal bias-variance tradeoff).

    Returns
    -------
    LocalWhittleResult

    References
    ----------
    Robinson 1995, "Gaussian semiparametric estimation of long-range
    dependence."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="localwhittle")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 200:
        raise KuantValueError(
            f"kuant.localwhittle: only {n} finite values; need at least "
            f"200.  [KE-VAL-MIN-CLEAN]"
        )
    if m is None:
        m = int(round(n**0.7))
    require_range(m, "m", kernel="localwhittle", lo=10, hi=n // 2)

    # Demean, compute periodogram.
    arr = arr - arr.mean()
    fft_vals = np.fft.fft(arr)
    I = np.abs(fft_vals) ** 2 / (2 * np.pi * n)  # noqa: E741 - I(λ) is standard periodogram notation
    # Fourier frequencies (positive half).
    freqs = 2 * np.pi * np.arange(1, n // 2 + 1) / n
    I_pos = I[1 : n // 2 + 1]

    m_eff = min(int(m), I_pos.size)
    w = freqs[:m_eff]
    Ij = I_pos[:m_eff]
    log_w = np.log(w)
    mean_log_w = float(np.mean(log_w))

    # Local Whittle objective.
    def neg_ll(d):
        # G(d) = mean(w_j^(2d) * I_j) - the local variance estimate.
        G = float(np.mean((w ** (2 * d)) * Ij))
        if G <= 0:
            return np.inf
        return np.log(G) - 2 * d * mean_log_w

    # Scan over a coarse grid, then refine via Brent-style bisection.
    grid = np.linspace(-0.49, 0.99, 149)
    vals = np.array([neg_ll(d) for d in grid])
    best = int(np.argmin(vals))
    lo = grid[max(best - 2, 0)]
    hi = grid[min(best + 2, len(grid) - 1)]
    # Refined golden-section.
    phi = (np.sqrt(5.0) - 1) / 2
    for _ in range(40):
        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)
        if neg_ll(c) < neg_ll(d):
            hi = d
        else:
            lo = c
    d_hat = 0.5 * (lo + hi)
    # Asymptotic SE.
    se = 0.5 / np.sqrt(m_eff)
    return LocalWhittleResult(
        d=float(d_hat),
        hurst=float(d_hat + 0.5),
        m=int(m_eff),
        n=int(n),
        se=float(se),
    )


__all__ = ["LocalWhittleResult", "localwhittle"]

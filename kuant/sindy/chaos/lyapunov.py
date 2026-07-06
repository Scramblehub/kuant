"""Largest Lyapunov exponent via the Rosenstein-Collins-DeLuca algorithm.

The largest Lyapunov exponent λ_1 measures the exponential rate at
which nearby trajectories diverge in phase space:

    |δ(t)| ≈ |δ(0)| · exp(λ_1 · t)

λ_1 > 0 is the fingerprint of chaos. λ_1 ≈ 0 is a periodic or
quasi-periodic regime. λ_1 < 0 is a fixed point (converges).

Rosenstein 1993 is the go-to method for small-to-medium time series
(hundreds to tens of thousands of observations), which is what most
financial signals look like. For very long noisy series, Kantz's
neighborhood method is more robust; consider it a v1.1 addition.

Design: docs/kernels/sindy/chaos/lyapunov.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
from kuant.errors import KuantValueError
from kuant.sindy.chaos.embedding import _embed


@dataclass
class LyapunovResult:
    """Rosenstein Lyapunov estimate.

    Attributes
    ----------
    lyapunov : float
        Largest Lyapunov exponent (nats per sample). Positive means
        divergent (chaotic-like); zero or negative means stable.
    intercept : float
        Intercept of the linear fit on the log-divergence curve.
    slope_range : tuple[int, int]
        The (start, end) sample indices over which the linear fit was
        run. Should cover the "linear" part of the divergence curve.
    log_divergence : 1D np.ndarray
        Mean log-divergence at each time step. Full curve returned so
        callers can visually check the fit region.
    embed_dim : int
    embed_tau : int
    """

    lyapunov: float
    intercept: float
    slope_range: tuple
    log_divergence: np.ndarray
    embed_dim: int
    embed_tau: int

    def summary(self) -> str:
        return (
            "=== LyapunovResult ===\n"
            f"lambda (nats/sample):  {self.lyapunov:+.6f}\n"
            f"embed dim / tau:        {self.embed_dim} / {self.embed_tau}\n"
            f"fit range:              [{self.slope_range[0]}, "
            f"{self.slope_range[1]}]\n"
            f"curve length:           {self.log_divergence.size}"
        )


def lyapunov(
    x,
    *,
    tau: int = 1,
    m: int = 5,
    max_t: int | None = None,
    theiler_window: int | None = None,
    fit_start: int = 1,
    fit_end: int | None = None,
) -> LyapunovResult:
    """Largest Lyapunov exponent via Rosenstein-Collins-DeLuca 1993.

    Parameters
    ----------
    x : 1D array
    tau : int, default 1
        Embedding delay. Use `mutualinfo(x).suggested_tau` if unsure.
    m : int, default 5
        Embedding dimension. Use `falsenearest(x, tau=tau).suggested_m`
        if unsure.
    max_t : int, optional
        Max number of time steps over which to track divergence.
        Defaults to `min(N // 4, 40)` where `N` is the embedded length.
    theiler_window : int, optional
        Minimum temporal separation between "nearest neighbors" (guards
        against picking a temporally-close point instead of a
        dynamically-close one). Defaults to `tau * m`.
    fit_start, fit_end : int
        Sample indices (in the log-divergence curve) over which to fit
        the slope. Rosenstein recommends fitting the initial linear
        rise; the default is `1` to `max_t // 2` which is a reasonable
        starting point.

    Returns
    -------
    LyapunovResult

    References
    ----------
    Rosenstein, Collins & DeLuca 1993, "A practical method for
    calculating largest Lyapunov exponents from small data sets."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="lyapunov")
    finite = np.isfinite(arr)
    arr = arr[finite]
    require_positive(tau, "tau", kernel="lyapunov", kind="int")
    require_range(m, "m", kernel="lyapunov", lo=2, hi=50)
    if arr.size < 200:
        raise KuantValueError(
            f"kuant.lyapunov: only {arr.size} finite values; need at "
            f"least 200 for a stable Rosenstein estimate.  "
            f"[KE-VAL-MIN-CLEAN]"
        )

    E = _embed(arr, int(m), int(tau))  # shape (N, m)
    N = E.shape[0]
    if max_t is None:
        max_t = int(min(N // 4, 40))
    if theiler_window is None:
        theiler_window = int(m) * int(tau)
    if fit_end is None:
        fit_end = max(fit_start + 2, max_t // 2)
    require_positive(max_t, "max_t", kernel="lyapunov", kind="int")

    # For each point i, find its nearest neighbor j that is at least
    # `theiler_window` samples away in time.
    nn_idx = np.full(N, -1, dtype=np.int64)
    for i in range(N):
        # Compute squared distance to all j; mask out temporally-close j.
        d2 = np.sum((E - E[i]) ** 2, axis=1)
        mask = np.abs(np.arange(N) - i) > theiler_window
        d2 = np.where(mask, d2, np.inf)
        nn_idx[i] = int(np.argmin(d2))

    # Track divergence over time.
    n_pairs = np.zeros(max_t, dtype=np.int64)
    log_div = np.zeros(max_t, dtype=np.float64)
    for k in range(max_t):
        # For every i whose neighbor j survives k steps forward, compute
        # log(distance between i+k and j+k).
        valid = (np.arange(N) + k < N) & (nn_idx + k < N)
        i_v = np.where(valid)[0]
        j_v = nn_idx[i_v]
        d = np.sqrt(np.sum((E[i_v + k] - E[j_v + k]) ** 2, axis=1))
        d = d[d > 0]
        if d.size == 0:
            continue
        log_div[k] = float(np.mean(np.log(d)))
        n_pairs[k] = d.size

    # Linear fit on [fit_start, fit_end].
    fit_end = min(fit_end, max_t - 1)
    if fit_end <= fit_start + 1:
        raise KuantValueError(
            f"kuant.lyapunov: fit range [{fit_start}, {fit_end}] is too "
            f"narrow; need at least 2 points.  [KE-VAL-RANGE]"
        )
    valid_curve = np.isfinite(log_div[fit_start : fit_end + 1])
    ks = np.arange(fit_start, fit_end + 1)[valid_curve]
    ys = log_div[fit_start : fit_end + 1][valid_curve]
    if ks.size < 2:
        raise KuantValueError(
            "kuant.lyapunov: could not compute divergence at enough "
            "steps to fit a slope.  [KE-VAL-MIN-CLEAN]"
        )
    slope, intercept = np.polyfit(ks, ys, 1)
    return LyapunovResult(
        lyapunov=float(slope),
        intercept=float(intercept),
        slope_range=(int(fit_start), int(fit_end)),
        log_divergence=log_div,
        embed_dim=int(m),
        embed_tau=int(tau),
    )


__all__ = ["LyapunovResult", "lyapunov"]

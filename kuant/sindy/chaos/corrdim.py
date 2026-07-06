"""Correlation dimension via the Grassberger-Procaccia algorithm.

The correlation dimension D_2 counts pair distances in a time-delay
embedding: for a well-chosen range of radii `r`, the fraction of
distinct pairs closer than `r` scales as `C(r) ~ r^D_2`. The
log-log slope of `C(r)` vs `r` is D_2.

D_2 is a low estimator for the box-counting dimension. Its
discriminant power in practice:

- Deterministic low-dim chaos: D_2 is a small finite non-integer
  (e.g. 2.05 for the Lorenz attractor).
- Periodic dynamics: D_2 = 1 (trajectory lies on a 1D closed curve).
- Stochastic noise: D_2 grows with the embedding dimension m (never
  saturates).

The saturation-vs-m check is the go-to test for low-dim chaos vs
noise: if D_2 keeps rising as m grows, the signal is stochastic.

Design: docs/kernels/sindy/chaos/corrdim.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
from kuant.errors import KuantValueError
from kuant.sindy.chaos.embedding import _embed


@dataclass
class CorrDimResult:
    """Grassberger-Procaccia correlation dimension estimate.

    Attributes
    ----------
    correlation_dim : float
        Slope of the log-log fit on the middle-range of C(r).
    log_r : 1D np.ndarray
        Log(r) values.
    log_C : 1D np.ndarray
        Log(C(r)) values.
    fit_range : tuple[int, int]
        (start, end) indices into log_r used for the linear fit.
    embed_dim : int
    embed_tau : int
    """

    correlation_dim: float
    log_r: np.ndarray
    log_C: np.ndarray
    fit_range: tuple
    embed_dim: int
    embed_tau: int

    def summary(self) -> str:
        return (
            "=== CorrDimResult ===\n"
            f"D_2:                {self.correlation_dim:.4f}\n"
            f"embed dim / tau:    {self.embed_dim} / {self.embed_tau}\n"
            f"fit range:          [{self.fit_range[0]}, {self.fit_range[1]}]\n"
            f"r grid points:      {self.log_r.size}"
        )


def corrdim(
    x,
    *,
    tau: int = 1,
    m: int = 5,
    n_r: int = 20,
    r_frac_range: tuple = (0.05, 0.5),
) -> CorrDimResult:
    """Grassberger-Procaccia correlation dimension.

    Parameters
    ----------
    x : 1D array
    tau : int, default 1
        Embedding delay.
    m : int, default 5
        Embedding dimension.
    n_r : int, default 20
        Number of radii to test along the log-spaced grid.
    r_frac_range : (float, float), default (0.05, 0.5)
        Radii range as fractions of the max pairwise distance. The
        middle of the log-log curve is where the scaling law holds;
        excluding the very small r (noise-dominated) and very large r
        (finite-size cutoff) is important for a stable slope.

    Returns
    -------
    CorrDimResult

    References
    ----------
    Grassberger & Procaccia 1983, "Characterization of strange
    attractors."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="corrdim")
    finite = np.isfinite(arr)
    arr = arr[finite]
    require_positive(tau, "tau", kernel="corrdim", kind="int")
    require_range(m, "m", kernel="corrdim", lo=2, hi=50)
    require_range(n_r, "n_r", kernel="corrdim", lo=5, hi=1000)
    if arr.size < 300:
        raise KuantValueError(
            f"kuant.corrdim: only {arr.size} finite values; need at "
            f"least 300 for a stable pairwise-distance estimate.  "
            f"[KE-VAL-MIN-CLEAN]"
        )
    lo, hi = float(r_frac_range[0]), float(r_frac_range[1])
    if not (0.0 < lo < hi < 1.0):
        raise KuantValueError(
            f"kuant.corrdim: 'r_frac_range' must satisfy 0 < lo < hi "
            f"< 1; got ({lo}, {hi}).  [KE-VAL-RANGE]"
        )

    E = _embed(arr, int(m), int(tau))
    # Pairwise squared distances (upper triangle only, no diagonal).
    N = E.shape[0]
    diff = E[:, None, :] - E[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    pair_d = d[iu]
    d_max = float(pair_d.max())
    r_grid = np.logspace(np.log10(lo * d_max), np.log10(hi * d_max), int(n_r))
    C = np.array([np.mean(pair_d < r) for r in r_grid])
    valid = C > 0
    log_r = np.log(r_grid[valid])
    log_C = np.log(C[valid])
    if log_r.size < 4:
        raise KuantValueError(
            "kuant.corrdim: fewer than 4 usable log-log points; "
            "widen 'r_frac_range' or provide more data.  "
            "[KE-VAL-MIN-CLEAN]"
        )
    # Fit on the middle 60% of the log-log curve.
    lo_i = int(0.2 * log_r.size)
    hi_i = int(0.8 * log_r.size)
    if hi_i - lo_i < 2:
        lo_i, hi_i = 0, log_r.size
    slope, _ = np.polyfit(log_r[lo_i:hi_i], log_C[lo_i:hi_i], 1)
    return CorrDimResult(
        correlation_dim=float(slope),
        log_r=log_r,
        log_C=log_C,
        fit_range=(int(lo_i), int(hi_i)),
        embed_dim=int(m),
        embed_tau=int(tau),
    )


__all__ = ["CorrDimResult", "corrdim"]

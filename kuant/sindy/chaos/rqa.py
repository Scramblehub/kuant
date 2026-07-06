"""Recurrence quantification analysis.

A recurrence plot (Eckmann-Kamphorst-Ruelle 1987) marks pairs of times
`(i, j)` at which a system's state was "close" in a time-delay
embedding, according to some radius `epsilon`. Structure in the plot
(diagonal lines, vertical lines) quantifies:

- **Recurrence rate (RR)**: fraction of recurrent pairs. Density of
  the plot.
- **Determinism (DET)**: fraction of recurrent pairs lying on
  diagonal lines of length >= `l_min`. High DET signals deterministic
  dynamics; near-zero DET signals stochastic noise.
- **Laminarity (LAM)**: same as DET but for vertical lines. Distinguishes
  chaotic transitions from laminar / trapped states.
- **Longest diagonal**: inverse of the divergence rate (proxy for the
  Lyapunov exponent).
- **Entropy of diagonal lengths**: Shannon entropy of the diagonal
  length distribution. Peaks for complex dynamics.

`rqa` returns all of these in a `RQAResult`.

Design: docs/kernels/sindy/chaos/rqa.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
from kuant.errors import KuantValueError
from kuant.sindy.chaos.embedding import _embed


@dataclass
class RQAResult:
    """Recurrence quantification measures.

    Attributes
    ----------
    recurrence_rate : float
        Density of the recurrence plot, in [0, 1].
    determinism : float
        Fraction of recurrent pairs in diagonals >= l_min.
    laminarity : float
        Fraction of recurrent pairs in verticals >= l_min.
    longest_diagonal : int
        Longest diagonal line length (excluding the main diagonal).
    entropy_diagonal : float
        Shannon entropy (nats) of the diagonal-length distribution.
    epsilon : float
        Radius used for the recurrence check.
    l_min : int
        Minimum diagonal / vertical length counted toward DET / LAM.
    embed_dim : int
    embed_tau : int
    """

    recurrence_rate: float
    determinism: float
    laminarity: float
    longest_diagonal: int
    entropy_diagonal: float
    epsilon: float
    l_min: int
    embed_dim: int
    embed_tau: int

    def summary(self) -> str:
        return (
            "=== RQAResult ===\n"
            f"recurrence rate:      {self.recurrence_rate:.4f}\n"
            f"determinism:          {self.determinism:.4f}\n"
            f"laminarity:           {self.laminarity:.4f}\n"
            f"longest diagonal:     {self.longest_diagonal}\n"
            f"diagonal-len entropy: {self.entropy_diagonal:.4f} nats\n"
            f"epsilon:              {self.epsilon:.4g}\n"
            f"l_min:                {self.l_min}\n"
            f"embed dim / tau:      {self.embed_dim} / {self.embed_tau}"
        )


def _diagonal_lengths(R: np.ndarray) -> list:
    """Lengths of contiguous diagonal-line segments in `R`, excluding the
    main diagonal (LOI)."""
    N = R.shape[0]
    lengths = []
    for k in range(1, N):
        # Diagonal offset by +k.
        diag = np.array([R[i, i + k] for i in range(N - k)])
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                run = 0
        if run > 0:
            lengths.append(run)
    return lengths


def _vertical_lengths(R: np.ndarray) -> list:
    """Lengths of contiguous vertical-line segments in `R`."""
    N = R.shape[0]
    lengths = []
    for j in range(N):
        col = R[:, j]
        run = 0
        for v in col:
            if v:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                run = 0
        if run > 0:
            lengths.append(run)
    return lengths


def rqa(
    x,
    *,
    tau: int = 1,
    m: int = 5,
    epsilon: float | None = None,
    recurrence_rate_target: float = 0.1,
    l_min: int = 2,
) -> RQAResult:
    """Recurrence quantification analysis.

    Parameters
    ----------
    x : 1D array
    tau : int, default 1
    m : int, default 5
    epsilon : float, optional
        Recurrence radius. If None, auto-set to reach
        `recurrence_rate_target` (typically 0.10).
    recurrence_rate_target : float, default 0.10
        Only used when `epsilon` is None. Kernel picks the epsilon
        whose recurrence rate is closest to this target.
    l_min : int, default 2
        Minimum diagonal / vertical length to count toward DET / LAM.

    Returns
    -------
    RQAResult

    References
    ----------
    Marwan, Romano, Thiel & Kurths 2007, "Recurrence plots for the
    analysis of complex systems."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="rqa")
    finite = np.isfinite(arr)
    arr = arr[finite]
    require_positive(tau, "tau", kernel="rqa", kind="int")
    require_range(m, "m", kernel="rqa", lo=2, hi=50)
    require_range(recurrence_rate_target, "recurrence_rate_target", kernel="rqa", lo=0.01, hi=0.9)
    require_positive(l_min, "l_min", kernel="rqa", kind="int")
    if arr.size < 100:
        raise KuantValueError(
            f"kuant.rqa: only {arr.size} finite values; need at least "
            f"100 for a stable recurrence plot.  [KE-VAL-MIN-CLEAN]"
        )
    if arr.size > 2000:
        # The recurrence matrix is O(N^2) in memory; cap for sanity.
        arr = arr[-2000:]

    E = _embed(arr, int(m), int(tau))
    N = E.shape[0]
    diff = E[:, None, :] - E[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))

    if epsilon is None:
        # Pick epsilon that hits the target rate.
        # Use a quantile of the off-diagonal distances.
        offdiag_d = d[np.triu_indices(N, k=1)]
        epsilon = float(np.quantile(offdiag_d, recurrence_rate_target))

    R = (d <= float(epsilon)).astype(np.int8)
    # Exclude the main diagonal (line-of-identity) from RR.
    np.fill_diagonal(R, 0)

    recurrence_rate = float(R.sum() / (N * N - N))
    diag_lengths = _diagonal_lengths(R)
    vert_lengths = _vertical_lengths(R)

    diag_long = [n for n in diag_lengths if n >= int(l_min)]
    vert_long = [n for n in vert_lengths if n >= int(l_min)]

    total_recurrent = int(R.sum())
    determinism = float(sum(diag_long)) / total_recurrent if total_recurrent > 0 else 0.0
    laminarity = float(sum(vert_long)) / total_recurrent if total_recurrent > 0 else 0.0
    longest_diag = int(max(diag_lengths)) if diag_lengths else 0

    if diag_long:
        counts = np.bincount(np.asarray(diag_long, dtype=np.int64))
        p = counts[counts > 0] / counts.sum()
        entropy_diag = float(-np.sum(p * np.log(p)))
    else:
        entropy_diag = 0.0

    return RQAResult(
        recurrence_rate=recurrence_rate,
        determinism=determinism,
        laminarity=laminarity,
        longest_diagonal=longest_diag,
        entropy_diagonal=entropy_diag,
        epsilon=float(epsilon),
        l_min=int(l_min),
        embed_dim=int(m),
        embed_tau=int(tau),
    )


__all__ = ["RQAResult", "rqa"]

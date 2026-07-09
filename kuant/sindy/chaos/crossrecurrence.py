"""Cross-recurrence and joint-recurrence quantification.

Extensions of the single-series `rqa` kernel to two coupled series:

- `crossrecurrence(x, y, ...)`: a cross-recurrence plot marks pairs
  `(i, j)` where the embedded state of `x` at time `i` is close to
  the embedded state of `y` at time `j`. Diagonal structure reveals
  synchronization; off-diagonal alignment reveals lagged coupling.
- `jointrecurrence(x, y, ...)`: joint-recurrence marks pairs `(i, j)`
  where BOTH `x` at `(i, j)` AND `y` at `(i, j)` are recurrent within
  their own state spaces. Highlights simultaneous recurrence events;
  used as a coupling / synchronization diagnostic distinct from cross
  recurrence.

Both return the standard RQA measures (recurrence rate, determinism,
laminarity, longest diagonal, entropy of diagonals) computed over the
appropriate recurrence matrix.

Design: docs/kernels/sindy/chaos/crossrecurrence.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
from kuant.errors import KuantValueError
from kuant.sindy.chaos.embedding import _embed


@dataclass
class CrossRecurrenceResult:
    """Cross-recurrence quantification measures.

    Attributes
    ----------
    recurrence_rate : float
    determinism : float
    laminarity : float
    longest_diagonal : int
    entropy_diagonal : float
    epsilon : float
    l_min : int
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
            "=== CrossRecurrenceResult ===\n"
            f"recurrence rate:      {self.recurrence_rate:.4f}\n"
            f"determinism:          {self.determinism:.4f}\n"
            f"laminarity:           {self.laminarity:.4f}\n"
            f"longest diagonal:     {self.longest_diagonal}\n"
            f"diag-len entropy:     {self.entropy_diagonal:.4f} nats\n"
            f"epsilon:              {self.epsilon:.4g}\n"
            f"l_min:                {self.l_min}\n"
            f"embed dim/tau:        {self.embed_dim} / {self.embed_tau}"
        )


@dataclass
class JointRecurrenceResult:
    """Joint-recurrence quantification measures. Same fields as
    CrossRecurrenceResult (see there)."""

    recurrence_rate: float
    determinism: float
    laminarity: float
    longest_diagonal: int
    entropy_diagonal: float
    epsilon_x: float
    epsilon_y: float
    l_min: int
    embed_dim: int
    embed_tau: int

    def summary(self) -> str:
        return (
            "=== JointRecurrenceResult ===\n"
            f"recurrence rate:      {self.recurrence_rate:.4f}\n"
            f"determinism:          {self.determinism:.4f}\n"
            f"laminarity:           {self.laminarity:.4f}\n"
            f"longest diagonal:     {self.longest_diagonal}\n"
            f"diag-len entropy:     {self.entropy_diagonal:.4f} nats\n"
            f"epsilon x / y:        {self.epsilon_x:.4g} / {self.epsilon_y:.4g}\n"
            f"l_min:                {self.l_min}\n"
            f"embed dim/tau:        {self.embed_dim} / {self.embed_tau}"
        )


def _diagonal_lengths(r_mat: np.ndarray, exclude_loi: bool) -> list:
    """Diagonal run lengths across BOTH sub- and super-diagonals.

    The cross-recurrence matrix is asymmetric (R[i,j] = 1 iff
    dist(x_i, y_j) <= eps), so line-structure lives on both sides of
    the main diagonal. Walking only k >= 0 (upper triangle) would
    halve determinism / longest-diagonal / diag entropy for CRP. For
    the symmetric jointrecurrence matrix, upper- and lower-triangle
    diagonals mirror each other, so we scan both here and let the
    caller's LOI exclusion (main diagonal) do the right thing.
    """
    n = r_mat.shape[0]
    lengths = []
    for k in range(-(n - 1), n):
        if exclude_loi and k == 0:
            continue
        # Diagonal at offset k has length (n - |k|).
        if k >= 0:
            diag = np.array([r_mat[i, i + k] for i in range(n - k)])
        else:
            diag = np.array([r_mat[i - k, i] for i in range(n + k)])
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


def _vertical_lengths(r_mat: np.ndarray) -> list:
    n = r_mat.shape[0]
    lengths = []
    for j in range(n):
        col = r_mat[:, j]
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


def _rqa_from_matrix(r_mat, l_min, exclude_loi):
    n = r_mat.shape[0]
    if exclude_loi:
        denom = n * n - n
    else:
        denom = n * n
    rr = float(r_mat.sum() / denom) if denom > 0 else 0.0
    diag_lengths = _diagonal_lengths(r_mat, exclude_loi=exclude_loi)
    vert_lengths = _vertical_lengths(r_mat)
    diag_long = [d for d in diag_lengths if d >= int(l_min)]
    vert_long = [d for d in vert_lengths if d >= int(l_min)]
    total_recurrent = int(r_mat.sum())
    det = float(sum(diag_long)) / total_recurrent if total_recurrent > 0 else 0.0
    lam = float(sum(vert_long)) / total_recurrent if total_recurrent > 0 else 0.0
    longest = int(max(diag_lengths)) if diag_lengths else 0
    if diag_long:
        counts = np.bincount(np.asarray(diag_long, dtype=np.int64))
        p = counts[counts > 0] / counts.sum()
        ent = float(-np.sum(p * np.log(p)))
    else:
        ent = 0.0
    return rr, det, lam, longest, ent


def crossrecurrence(
    x,
    y,
    *,
    tau: int = 1,
    m: int = 5,
    epsilon: float | None = None,
    recurrence_rate_target: float = 0.1,
    l_min: int = 2,
) -> CrossRecurrenceResult:
    """Cross-recurrence quantification between series `x` and `y`.

    Parameters
    ----------
    x, y : 1D arrays of equal length
    tau, m : embedding parameters (same for both series)
    epsilon : float, optional
        Distance threshold. If None, picked from a quantile of cross-
        pair distances to hit `recurrence_rate_target`.
    recurrence_rate_target : float, default 0.10
    l_min : int, default 2

    Returns
    -------
    CrossRecurrenceResult

    References
    ----------
    Marwan et al 2007, extends single-series RQA to two series.
    """
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    require_1d(arr_x, "x", kernel="crossrecurrence")
    require_1d(arr_y, "y", kernel="crossrecurrence")
    if arr_x.size != arr_y.size:
        raise KuantValueError(
            f"kuant.crossrecurrence: 'x' and 'y' must be equal length; "
            f"got {arr_x.size} and {arr_y.size}.  [KE-SHAPE-EQUAL-LEN]"
        )
    require_positive(tau, "tau", kernel="crossrecurrence", kind="int")
    require_range(m, "m", kernel="crossrecurrence", lo=2, hi=50)
    require_range(
        recurrence_rate_target,
        "recurrence_rate_target",
        kernel="crossrecurrence",
        lo=0.01,
        hi=0.9,
    )
    require_positive(l_min, "l_min", kernel="crossrecurrence", kind="int")

    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    xf = arr_x[mask]
    yf = arr_y[mask]
    if xf.size < 100:
        raise KuantValueError(
            f"kuant.crossrecurrence: only {xf.size} paired finite values; "
            f"need at least 100.  [KE-VAL-MIN-CLEAN]"
        )
    if xf.size > 2000:
        xf = xf[-2000:]
        yf = yf[-2000:]

    Ex = _embed(xf, int(m), int(tau))
    Ey = _embed(yf, int(m), int(tau))
    # Cross-distance matrix (n_x by n_y). For equal-length series
    # after embedding they match.
    diff = Ex[:, None, :] - Ey[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))

    if epsilon is None:
        epsilon = float(np.quantile(d, recurrence_rate_target))

    r_mat = (d <= float(epsilon)).astype(np.int8)
    # Cross-recurrence does NOT have a line-of-identity by construction
    # (different series); do not exclude any diagonal.
    rr, det, lam, longest, ent = _rqa_from_matrix(r_mat, int(l_min), exclude_loi=False)
    return CrossRecurrenceResult(
        recurrence_rate=rr,
        determinism=det,
        laminarity=lam,
        longest_diagonal=longest,
        entropy_diagonal=ent,
        epsilon=float(epsilon),
        l_min=int(l_min),
        embed_dim=int(m),
        embed_tau=int(tau),
    )


def jointrecurrence(
    x,
    y,
    *,
    tau: int = 1,
    m: int = 5,
    epsilon_x: float | None = None,
    epsilon_y: float | None = None,
    recurrence_rate_target: float = 0.1,
    l_min: int = 2,
) -> JointRecurrenceResult:
    """Joint-recurrence quantification: intersect two single-series RPs.

    Parameters
    ----------
    x, y : 1D arrays of equal length
    tau, m : embedding parameters shared by both series
    epsilon_x, epsilon_y : float, optional
        Per-series thresholds. Each auto-picked from own-quantile to
        hit `recurrence_rate_target` if None.
    recurrence_rate_target : float, default 0.10
    l_min : int, default 2

    Returns
    -------
    JointRecurrenceResult

    References
    ----------
    Marwan et al 2007.
    """
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    require_1d(arr_x, "x", kernel="jointrecurrence")
    require_1d(arr_y, "y", kernel="jointrecurrence")
    if arr_x.size != arr_y.size:
        raise KuantValueError(
            f"kuant.jointrecurrence: 'x' and 'y' must be equal length; "
            f"got {arr_x.size} and {arr_y.size}.  [KE-SHAPE-EQUAL-LEN]"
        )
    require_positive(tau, "tau", kernel="jointrecurrence", kind="int")
    require_range(m, "m", kernel="jointrecurrence", lo=2, hi=50)
    require_range(
        recurrence_rate_target,
        "recurrence_rate_target",
        kernel="jointrecurrence",
        lo=0.01,
        hi=0.9,
    )
    require_positive(l_min, "l_min", kernel="jointrecurrence", kind="int")

    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    xf = arr_x[mask]
    yf = arr_y[mask]
    if xf.size < 100:
        raise KuantValueError(
            f"kuant.jointrecurrence: only {xf.size} paired finite values; "
            f"need at least 100.  [KE-VAL-MIN-CLEAN]"
        )
    if xf.size > 2000:
        xf = xf[-2000:]
        yf = yf[-2000:]

    Ex = _embed(xf, int(m), int(tau))
    Ey = _embed(yf, int(m), int(tau))

    # Own-series pairwise distances.
    def _pair(mat):
        diff = mat[:, None, :] - mat[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    dx = _pair(Ex)
    dy = _pair(Ey)
    if epsilon_x is None:
        offdiag_dx = dx[np.triu_indices(dx.shape[0], k=1)]
        epsilon_x = float(np.quantile(offdiag_dx, recurrence_rate_target))
    if epsilon_y is None:
        offdiag_dy = dy[np.triu_indices(dy.shape[0], k=1)]
        epsilon_y = float(np.quantile(offdiag_dy, recurrence_rate_target))

    r_x = (dx <= float(epsilon_x)).astype(np.int8)
    r_y = (dy <= float(epsilon_y)).astype(np.int8)
    np.fill_diagonal(r_x, 0)
    np.fill_diagonal(r_y, 0)
    r_mat = (r_x & r_y).astype(np.int8)

    rr, det, lam, longest, ent = _rqa_from_matrix(r_mat, int(l_min), exclude_loi=True)
    return JointRecurrenceResult(
        recurrence_rate=rr,
        determinism=det,
        laminarity=lam,
        longest_diagonal=longest,
        entropy_diagonal=ent,
        epsilon_x=float(epsilon_x),
        epsilon_y=float(epsilon_y),
        l_min=int(l_min),
        embed_dim=int(m),
        embed_tau=int(tau),
    )


__all__ = [
    "CrossRecurrenceResult",
    "JointRecurrenceResult",
    "crossrecurrence",
    "jointrecurrence",
]

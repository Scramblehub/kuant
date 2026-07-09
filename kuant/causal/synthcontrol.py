"""Synthetic Control Method (Abadie, Diamond, Hainmueller 2010).

Constructs a "synthetic" version of a treated unit as a convex
combination of untreated donor units, weighted to match pre-treatment
outcomes. Post-treatment gap between treated and synthetic = treatment
effect estimate.

Weights w >= 0, sum(w) == 1. Solved by projected gradient descent on
the pre-period MSE, restricted to the simplex.

Standard use case: policy analysis, natural experiments, event studies
where random assignment is unavailable and one treated unit has many
plausible controls (states, firms, countries).

Sign convention: `att` (average treatment effect on the treated) is
positive when treated OUTPERFORMS its synthetic counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_positive, require_range
from kuant.errors import KuantValueError


@dataclass
class SynthControlResult:
    weights: np.ndarray
    treated_pre_fit_rmse: float
    treated_post: np.ndarray
    synthetic_post: np.ndarray
    gap_post: np.ndarray
    att: float
    n_donors: int
    t_pre: int
    t_post: int

    def summary(self) -> str:
        top_idx = np.argsort(self.weights)[::-1][:5]
        top_str = ", ".join(
            f"[{i}]={self.weights[i]:.3f}" for i in top_idx if self.weights[i] > 1e-4
        )
        return (
            "=== SynthControlResult ===\n"
            f"ATT (post-period):    {self.att:+.6f}\n"
            f"pre-fit RMSE:         {self.treated_pre_fit_rmse:.6f}\n"
            f"donors used:          {int((self.weights > 1e-4).sum())} / {self.n_donors}\n"
            f"top weights:          {top_str}\n"
            f"pre / post periods:   {self.t_pre} / {self.t_post}"
        )


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def synthcontrol(
    treated,
    donors,
    t_treat: int,
    *,
    n_iter: int = 2000,
    tol: float = 1e-8,
) -> SynthControlResult:
    """Abadie-Diamond-Hainmueller synthetic control.

    Parameters
    ----------
    treated : 1D array of length T
        Outcome series for the treated unit.
    donors : 2D array of shape (T, J)
        Outcome series for J candidate donor units.
    t_treat : int
        First post-treatment index (0-indexed). treated[:t_treat] is
        pre-period; treated[t_treat:] is post-period.
    n_iter : int, default 2000
    tol : float, default 1e-8
        Convergence criterion on the objective delta.

    Returns
    -------
    SynthControlResult

    References
    ----------
    Abadie, Diamond & Hainmueller 2010, "Synthetic Control Methods for
    Comparative Case Studies." JASA. Abadie 2021 review in JEL.
    """
    y = np.asarray(treated, dtype=np.float64).reshape(-1)
    X = np.asarray(donors, dtype=np.float64)
    if X.ndim != 2:
        raise KuantValueError(
            f"kuant.synthcontrol: 'donors' must be 2D (T x J); got "
            f"shape {X.shape}.  [KE-SHAPE-2D]"
        )
    T, J = X.shape
    if y.size != T:
        raise KuantValueError(
            f"kuant.synthcontrol: 'treated' length ({y.size}) must match "
            f"donors rows ({T}).  [KE-SHAPE-EQUAL-LEN]"
        )
    require_range(t_treat, "t_treat", kernel="synthcontrol", lo=2, hi=T - 1)
    require_positive(n_iter, "n_iter", kernel="synthcontrol", kind="int")

    y_pre = y[:t_treat]
    X_pre = X[:t_treat, :]
    y_post = y[t_treat:]
    X_post = X[t_treat:, :]

    mask_finite = np.isfinite(y_pre) & np.all(np.isfinite(X_pre), axis=1)
    if mask_finite.sum() < 3:
        raise KuantValueError(
            "kuant.synthcontrol: fewer than 3 clean pre-treatment "
            "observations.  [KE-VAL-MIN-CLEAN]"
        )
    y_pre_c = y_pre[mask_finite]
    X_pre_c = X_pre[mask_finite, :]

    # Scale for stable step size.
    scale = float(np.std(y_pre_c) + 1e-12)
    y_pre_s = y_pre_c / scale
    X_pre_s = X_pre_c / scale

    w = np.full(J, 1.0 / J, dtype=np.float64)
    lr = 1.0 / (np.linalg.norm(X_pre_s, ord=2) ** 2 + 1e-12)
    prev_loss = np.inf
    for _ in range(int(n_iter)):
        resid = X_pre_s @ w - y_pre_s
        grad = X_pre_s.T @ resid / X_pre_s.shape[0]
        w = _project_simplex(w - lr * grad)
        loss = 0.5 * float(np.mean(resid**2))
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    synth_pre = X_pre_c @ w
    rmse_pre = float(np.sqrt(np.mean((synth_pre - y_pre_c) ** 2)))
    synth_post = X_post @ w
    gap_post = y_post - synth_post
    att = float(np.mean(gap_post[np.isfinite(gap_post)]))

    return SynthControlResult(
        weights=w,
        treated_pre_fit_rmse=rmse_pre,
        treated_post=y_post,
        synthetic_post=synth_post,
        gap_post=gap_post,
        att=att,
        n_donors=int(J),
        t_pre=int(t_treat),
        t_post=int(T - t_treat),
    )


__all__ = ["SynthControlResult", "synthcontrol"]

"""Risk parity (equal risk contribution) portfolio.

Weights satisfying: each asset contributes an equal share of portfolio
variance. Formally, `w_i * (Sigma w)_i / (w' Sigma w) = 1/n` for all i.

Solved iteratively via coordinate descent on the log-barrier form; the
objective is convex in log(w) so convergence is well-behaved for
positive-definite covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class RiskParityResult:
    weights: np.ndarray
    risk_contributions: np.ndarray
    portfolio_variance: float
    n_iters: int
    converged: bool

    def summary(self) -> str:
        return (
            "=== RiskParityResult ===\n"
            f"n assets:            {self.weights.size}\n"
            f"portfolio variance:  {self.portfolio_variance:.6f}\n"
            f"weight sum:          {self.weights.sum():.4f}\n"
            f"max risk contrib:    {self.risk_contributions.max():.4f}\n"
            f"min risk contrib:    {self.risk_contributions.min():.4f}\n"
            f"converged:           {self.converged} in {self.n_iters} iters"
        )


def riskparity(
    cov,
    *,
    target=None,
    max_iters: int = 500,
    tol: float = 1e-8,
) -> RiskParityResult:
    """Equal risk contribution (ERC) portfolio via cyclic coordinate descent.

    Parameters
    ----------
    cov : 2D array, shape (n, n)
        Asset covariance matrix.
    target : 1D array, length n, optional
        Target risk contributions (normalized to sum to 1). If None,
        equal-risk (1/n each) is used.
    max_iters : int, default 500
    tol : float, default 1e-8

    Returns
    -------
    RiskParityResult

    References
    ----------
    Maillard, Roncalli & Teiletche 2010, "The properties of equally
    weighted risk contribution portfolios."
    """
    Sigma = np.asarray(cov, dtype=np.float64)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise KuantShapeError(
            f"kuant.riskparity: 'cov' must be square; got shape " f"{Sigma.shape}.  [KE-SHAPE-2D]"
        )
    n = Sigma.shape[0]
    require_positive(max_iters, "max_iters", kernel="riskparity", kind="int")
    require_positive(tol, "tol", kernel="riskparity")

    if target is None:
        b = np.ones(n) / n
    else:
        b = np.asarray(target, dtype=np.float64).ravel()
        if b.size != n:
            raise KuantShapeError(
                f"kuant.riskparity: target size {b.size} does not match "
                f"cov dim {n}.  [KE-SHAPE-EQUAL-LEN]"
            )
        if (b <= 0).any():
            raise KuantValueError(
                "kuant.riskparity: target contributions must be strictly "
                "positive.  [KE-VAL-POSITIVE]"
            )
        b = b / b.sum()

    # Coordinate descent (Maillard-Roncalli-Teiletche 2010).
    w = np.ones(n) / n
    converged = False
    for it in range(int(max_iters)):
        Sw = Sigma @ w
        rc = w * Sw
        total = float(rc.sum())
        target_rc = b * total
        # Update: for each i, w_i * (Sigma w)_i = b_i * (w' Sigma w).
        # Cyclic update via square root of the ratio.
        max_change = 0.0
        for i in range(n):
            denom = float(Sigma[i, i])
            if denom <= 0:
                continue
            # Marginal (Sigma w)_i excluding own contribution.
            partial = float(Sw[i] - Sigma[i, i] * w[i])
            # Solve w_i^2 * denom + w_i * partial - target_rc[i] = 0.
            # Positive root.
            disc = partial * partial + 4.0 * denom * target_rc[i]
            if disc < 0:
                continue
            new_w = (-partial + np.sqrt(disc)) / (2.0 * denom)
            change = abs(new_w - w[i])
            if change > max_change:
                max_change = change
            w[i] = new_w
            # Update Sw efficiently.
            Sw = Sigma @ w
        w = w / w.sum() if w.sum() > 0 else w
        if max_change < tol:
            converged = True
            break

    port_var = float(w @ Sigma @ w)
    rc_final = w * (Sigma @ w) / port_var if port_var > 0 else np.zeros(n)
    return RiskParityResult(
        weights=w,
        risk_contributions=rc_final,
        portfolio_variance=port_var,
        n_iters=int(it + 1),
        converged=converged,
    )


__all__ = ["RiskParityResult", "riskparity"]

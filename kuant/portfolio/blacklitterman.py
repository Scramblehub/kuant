"""Black-Litterman posterior mean-variance portfolio (1990).

Combines an equilibrium prior on expected returns with investor views
to produce a shrinkage-adjusted posterior. Under Gaussian assumptions:

    posterior_mean = [(tau*Sigma)^-1 + P' * Omega^-1 * P]^-1
                   * [(tau*Sigma)^-1 * pi + P' * Omega^-1 * Q]
    posterior_cov  = Sigma + [(tau*Sigma)^-1 + P' * Omega^-1 * P]^-1

Optimal weights follow from the standard mean-variance:
    w = (1/lambda) * posterior_cov^-1 * posterior_mean

with `posterior_mean` interpreted as EXCESS returns over the risk-free
rate (the standard Black-Litterman convention; subtract rf from prior_mean
before calling if working in raw returns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantShapeError


@dataclass
class BlackLittermanResult:
    posterior_mean: np.ndarray
    posterior_cov: np.ndarray
    weights: np.ndarray
    prior_mean: np.ndarray
    views_shift: np.ndarray  # posterior_mean - prior_mean

    def summary(self) -> str:
        return (
            "=== BlackLittermanResult ===\n"
            f"posterior mean:  {self.posterior_mean}\n"
            f"weights:         {self.weights}\n"
            f"view shift:      {self.views_shift}"
        )


def blacklitterman(
    prior_mean,
    prior_cov,
    P,
    Q,
    *,
    Omega=None,
    tau: float = 0.05,
    risk_aversion: float = 3.0,
) -> BlackLittermanResult:
    """Black-Litterman posterior + optimal weights.

    Parameters
    ----------
    prior_mean : 1D array, length n_assets
        Equilibrium (or CAPM) expected returns.
    prior_cov : 2D array, shape (n_assets, n_assets)
    P : 2D array, shape (n_views, n_assets)
        Views matrix. Each row picks the assets involved in one view.
    Q : 1D array, length n_views
        View expected returns.
    Omega : 2D array, shape (n_views, n_views), optional
        View covariance. If None, defaults to `tau * P * prior_cov * P'`
        (Idzorek proportional method).
    tau : float, default 0.05
        Uncertainty scaling on the prior.
    risk_aversion : float, default 3.0
        Optimal weights lambda.

    Returns
    -------
    BlackLittermanResult

    References
    ----------
    Black & Litterman 1990. Idzorek 2005 for the proportional Omega.
    """
    pi = np.asarray(prior_mean, dtype=np.float64).ravel()
    Sigma = np.asarray(prior_cov, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64).ravel()
    require_positive(tau, "tau", kernel="blacklitterman")
    require_positive(risk_aversion, "risk_aversion", kernel="blacklitterman")

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1] or Sigma.shape[0] != pi.size:
        raise KuantShapeError(
            f"kuant.blacklitterman: prior_cov must be (n, n) matching prior_mean; "
            f"got prior_mean size {pi.size} and prior_cov shape {Sigma.shape}.  "
            f"[KE-SHAPE-2D]"
        )
    if P.ndim != 2 or P.shape[1] != pi.size:
        raise KuantShapeError(
            f"kuant.blacklitterman: P must be (n_views, {pi.size}); got shape "
            f"{P.shape}.  [KE-SHAPE-2D]"
        )
    if Q.size != P.shape[0]:
        raise KuantShapeError(
            f"kuant.blacklitterman: Q length ({Q.size}) must match P row "
            f"count ({P.shape[0]}).  [KE-SHAPE-EQUAL-LEN]"
        )

    if Omega is None:
        # Idzorek proportional Omega.
        Omega = float(tau) * (P @ Sigma @ P.T)
        # Ensure positive-definiteness for near-singular cases.
        Omega = Omega + 1e-10 * np.eye(P.shape[0])
    else:
        Omega = np.asarray(Omega, dtype=np.float64)
        if Omega.shape != (P.shape[0], P.shape[0]):
            raise KuantShapeError(
                f"kuant.blacklitterman: Omega must be ({P.shape[0]}, "
                f"{P.shape[0]}); got {Omega.shape}.  [KE-SHAPE-2D]"
            )

    tau_Sigma_inv = np.linalg.inv(float(tau) * Sigma)
    Omega_inv = np.linalg.inv(Omega)
    Mprec = tau_Sigma_inv + P.T @ Omega_inv @ P
    Mcov = np.linalg.inv(Mprec)
    posterior_mean = Mcov @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
    posterior_cov = Sigma + Mcov

    # Optimal weights (unconstrained mean-variance).
    weights = np.linalg.solve(float(risk_aversion) * posterior_cov, posterior_mean)

    return BlackLittermanResult(
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        weights=weights,
        prior_mean=pi,
        views_shift=posterior_mean - pi,
    )


__all__ = ["BlackLittermanResult", "blacklitterman"]

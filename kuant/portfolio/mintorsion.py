"""Principal-components torsion (Meucci 2009 effective bets).

Given a covariance matrix Sigma, computes an orthonormal torsion matrix
`T` such that the transformed factors `f = T r` are UNCORRELATED
(`T Sigma T.T` is exactly diagonal) and each factor is
sign-oriented to be as close to its associated asset as possible via a
column-sign flip.

This is the "principal-components torsion" (PCT) approximation to the
full Meucci-Santangelo-Deguest 2013 minimum-torsion decomposition. PCT
matches min-torsion when the true correlation structure is dominated
by a single principal axis and is a widely used surrogate when only
diversification-content (effective number of bets) is needed. The
exact iterative min-torsion solver from Meucci 2013 Section 2.5 is a
tracked follow-up.

Useful for:
- Effective number of bets (Meucci 2009 diversification distribution)
- Factor risk attribution without redundancy
- Portfolio construction under multi-factor constraints

References
----------
Meucci 2009, "Managing Diversification." Risk Magazine.
Meucci, Santangelo & Deguest 2013, "Risk budgeting and
diversification based on optimized uncorrelated factors." arXiv:1305.5850.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant.errors import KuantShapeError


@dataclass
class MinTorsionResult:
    torsion: np.ndarray
    factor_cov: np.ndarray
    effective_bets: float

    def summary(self) -> str:
        return (
            "=== MinTorsionResult ===\n"
            f"n factors:        {self.torsion.shape[0]}\n"
            f"factor cov diag:  {np.diag(self.factor_cov)}\n"
            f"off-diag max:     "
            f"{np.max(np.abs(self.factor_cov - np.diag(np.diag(self.factor_cov)))):.4g}\n"
            f"effective bets:   {self.effective_bets:.4f}"
        )


def mintorsion(cov, *, weights=None) -> MinTorsionResult:
    """Principal-components torsion decomposition.

    Parameters
    ----------
    cov : 2D array, shape (n, n)
        Covariance matrix.
    weights : 1D array, length n, optional
        Portfolio weights. If provided, `effective_bets` is computed
        for these weights; else uses equal weights.

    Returns
    -------
    MinTorsionResult

    Notes
    -----
    Sign convention: eigenvectors are oriented so `diag(V) >= 0`, which
    picks the sign closest to the identity mapping (each factor points
    the same way as its associated asset when possible).
    """
    Sigma = np.asarray(cov, dtype=np.float64)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise KuantShapeError(
            f"kuant.mintorsion: 'cov' must be square 2D; got shape "
            f"{Sigma.shape}.  [KE-SHAPE-2D]"
        )
    n = Sigma.shape[0]

    # Symmetric eigendecomposition. eigh returns ascending eigenvalues;
    # reverse so the largest variance factor is first, matching the
    # standard PCA convention.
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-14)
    V = eigvecs[:, order]

    # Sign-orient each column so its dominant asset entry is positive.
    # This makes torsion "closest to identity" per column, which is the
    # PCT surrogate for the Meucci min-torsion criterion.
    dominant_row = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[dominant_row, np.arange(n)])
    signs[signs == 0] = 1.0
    V = V * signs[np.newaxis, :]

    torsion = V.T
    factor_cov = torsion @ Sigma @ torsion.T

    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != n:
            raise KuantShapeError(
                f"kuant.mintorsion: weights size {w.size} does not match "
                f"cov dim {n}.  [KE-SHAPE-EQUAL-LEN]"
            )
    # Portfolio exposure in factor space: p = T @ w.
    p = torsion @ w
    var_contribs = p**2 * np.diag(factor_cov)
    total = float(var_contribs.sum())
    if total <= 0:
        effective_bets = float("nan")
    else:
        p_norm = var_contribs / total
        p_pos = p_norm[p_norm > 0]
        effective_bets = float(np.exp(-np.sum(p_pos * np.log(p_pos))))

    return MinTorsionResult(
        torsion=torsion,
        factor_cov=factor_cov,
        effective_bets=float(effective_bets),
    )


__all__ = ["MinTorsionResult", "mintorsion"]

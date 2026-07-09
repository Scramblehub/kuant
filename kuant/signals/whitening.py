"""Whitening transforms: ZCA and PCA whitening.

Both produce a linear transformation `X_white = X @ W` such that
`cov(X_white) = I` (identity). They differ in the choice of rotation:

- PCA whitening: `W = V * diag(1/sqrt(lambda))` where V, lambda are
  eigenvectors and eigenvalues of `X.T @ X / n`. Rotates into
  principal components AND scales.
- ZCA whitening: `W = V * diag(1/sqrt(lambda)) * V.T`. Adds a rotation
  back to preserve original feature interpretation. Best "closest to
  identity" whitening in Frobenius norm.

Both are numerically robust via a small ridge regularizer on the
eigenvalues to handle rank-deficient matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class WhiteningResult:
    X_white: np.ndarray
    W: np.ndarray
    eigenvalues: np.ndarray
    mean: np.ndarray
    method: str

    def summary(self) -> str:
        return (
            f"=== WhiteningResult ({self.method}) ===\n"
            f"input shape:      {self.X_white.shape}\n"
            f"eigenvalues:      {np.round(self.eigenvalues, 4)}\n"
            f"empirical cov trace / d: "
            f"{float(np.trace(self.X_white.T @ self.X_white) / (self.X_white.shape[0] * self.X_white.shape[1])):.4f}"
        )


def whitening(
    X,
    *,
    method: str = "zca",
    ridge: float = 1e-6,
) -> WhiteningResult:
    """Whiten a data matrix so that its columns become uncorrelated.

    Parameters
    ----------
    X : 2D array, shape (n, d)
    method : {"zca", "pca"}, default "zca"
    ridge : float, default 1e-6
        Regularizer added to eigenvalues before inversion.

    Returns
    -------
    WhiteningResult

    References
    ----------
    Kessy, Lewin & Strimmer 2015, "Optimal whitening and decorrelation."
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise KuantShapeError(
            f"kuant.whitening: 'X' must be 2D; got shape {Xa.shape}.  " f"[KE-SHAPE-EXPECTED]"
        )
    if method not in ("zca", "pca"):
        raise KuantValueError(
            f"kuant.whitening: 'method' must be 'zca' or 'pca', got " f"{method!r}.  [KE-VAL-RANGE]"
        )
    require_positive(ridge, "ridge", kernel="whitening")

    n, d = Xa.shape
    if n < d + 5:
        raise KuantValueError(
            f"kuant.whitening: {n} rows and {d} columns; need at least "
            f"d + 5 rows.  [KE-VAL-MIN-CLEAN]"
        )
    mu = Xa.mean(axis=0)
    Xc = Xa - mu
    C = (Xc.T @ Xc) / n
    eigvals, V = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, float(ridge), None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    if method == "pca":
        W = V @ D_inv_sqrt
    else:  # zca
        W = V @ D_inv_sqrt @ V.T
    X_white = Xc @ W
    return WhiteningResult(
        X_white=X_white,
        W=W,
        eigenvalues=eigvals,
        mean=mu,
        method=method,
    )


__all__ = ["WhiteningResult", "whitening"]

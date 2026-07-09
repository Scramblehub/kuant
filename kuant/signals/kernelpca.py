"""Kernel Principal Component Analysis (kernel PCA).

Nonlinear dimensionality reduction via the kernel trick. Standard
kernels: RBF (Gaussian), polynomial, sigmoid. Wrapper around
scikit-learn's KernelPCA.

When PCA on raw features misses nonlinear structure (e.g., clusters
lying on curved manifolds), kernel PCA can recover it. Common uses:
- Regime detection on nonlinear return-embedding spaces.
- Denoising with an RBF kernel followed by inverse transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class KernelPcaResult:
    components: np.ndarray  # (n, k) transformed data
    eigenvalues: np.ndarray
    kernel: str

    def summary(self) -> str:
        return (
            "=== KernelPcaResult ===\n"
            f"components shape: {self.components.shape}\n"
            f"eigenvalues:      {np.round(self.eigenvalues, 4)}\n"
            f"kernel:           {self.kernel}"
        )


def kernelpca(
    X,
    *,
    n_components: int = 3,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 3,
) -> KernelPcaResult:
    """Kernel PCA.

    Parameters
    ----------
    X : 2D array, shape (n, d)
    n_components : int, default 3
    kernel : {"rbf", "poly", "sigmoid", "cosine", "linear"}, default "rbf"
    gamma : float, optional
        Kernel width. If None, uses 1/n_features (sklearn default).
    degree : int, default 3
        Polynomial degree (used only for "poly").

    Returns
    -------
    KernelPcaResult

    References
    ----------
    Scholkopf, Smola & Muller 1998, "Nonlinear component analysis as a
    kernel eigenvalue problem."
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise KuantShapeError(
            f"kuant.kernelpca: 'X' must be 2D; got shape {Xa.shape}.  " f"[KE-SHAPE-EXPECTED]"
        )
    n, d = Xa.shape
    if n_components < 1 or n_components > min(n, d):
        raise KuantValueError(
            f"kuant.kernelpca: 'n_components' must be in [1, {min(n, d)}]; "
            f"got {n_components}.  [KE-VAL-RANGE]"
        )
    if kernel not in ("rbf", "poly", "sigmoid", "cosine", "linear"):
        raise KuantValueError(
            f"kuant.kernelpca: 'kernel' must be one of "
            f"{{rbf, poly, sigmoid, cosine, linear}}, got {kernel!r}.  "
            f"[KE-VAL-RANGE]"
        )

    require_positive(degree, "degree", kernel="kernelpca", kind="int")

    try:
        from sklearn.decomposition import KernelPCA
    except ImportError as e:
        raise KuantValueError(
            "kuant.kernelpca: requires scikit-learn.  [KE-DEP-MISSING]\n"
            "  -> Fix: pip install scikit-learn"
        ) from e

    model = KernelPCA(
        n_components=int(n_components),
        kernel=str(kernel),
        gamma=gamma,
        degree=int(degree),
        eigen_solver="dense",
    )
    components = model.fit_transform(Xa)
    eigvals = np.asarray(getattr(model, "eigenvalues_", []), dtype=np.float64)
    return KernelPcaResult(
        components=components,
        eigenvalues=eigvals,
        kernel=str(kernel),
    )


__all__ = ["KernelPcaResult", "kernelpca"]

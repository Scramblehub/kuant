"""Independent Component Analysis (FastICA wrapper).

Separates a multivariate signal into statistically independent
components using FastICA (Hyvarinen 1999). Thin wrapper around
scikit-learn's FastICA under kuant's result-dataclass + error
hierarchy.

Financial use cases:
- Decomposing a factor return matrix into independent latent drivers.
- Signal separation for mixed alpha sources.
- Blind source separation on high-frequency price co-movements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import warnings

from kuant._validation import require_positive
from kuant.errors import KuantNumericWarning, KuantShapeError, KuantValueError


@dataclass
class IcaResult:
    sources: np.ndarray  # (n, k) recovered independent sources
    mixing: np.ndarray  # (d, k) estimated mixing matrix
    unmixing: np.ndarray  # (k, d) unmixing matrix
    mean: np.ndarray
    n_iters: int
    converged: bool

    def summary(self) -> str:
        return (
            "=== IcaResult ===\n"
            f"sources shape:    {self.sources.shape}\n"
            f"mixing shape:     {self.mixing.shape}\n"
            f"n iters:          {self.n_iters}\n"
            f"converged:        {self.converged}"
        )


def ica(
    X,
    *,
    n_components: int | None = None,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int | None = 0,
) -> IcaResult:
    """FastICA independent component analysis.

    Parameters
    ----------
    X : 2D array, shape (n, d)
    n_components : int, optional
        Number of independent sources to recover. Default d.
    max_iter : int, default 300
    tol : float, default 1e-4
    random_state : int, default 0

    Returns
    -------
    IcaResult

    References
    ----------
    Hyvarinen 1999, "Fast and robust fixed-point algorithms for
    independent component analysis."
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise KuantShapeError(
            f"kuant.ica: 'X' must be 2D; got shape {Xa.shape}.  " f"[KE-SHAPE-EXPECTED]"
        )
    require_positive(max_iter, "max_iter", kernel="ica", kind="int")
    require_positive(tol, "tol", kernel="ica")
    n, d = Xa.shape
    if n_components is None:
        n_components = d
    if n_components > d or n_components < 1:
        raise KuantValueError(
            f"kuant.ica: 'n_components' must be in [1, {d}]; got "
            f"{n_components}.  [KE-VAL-RANGE]"
        )

    try:
        from sklearn.decomposition import FastICA
    except ImportError as e:
        raise KuantValueError(
            "kuant.ica: requires scikit-learn.  [KE-DEP-MISSING]\n"
            "  -> Fix: pip install scikit-learn"
        ) from e

    model = FastICA(
        n_components=int(n_components),
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=random_state,
        whiten="unit-variance",
    )
    sources = model.fit_transform(Xa)
    mixing = model.mixing_
    unmixing = model.components_
    n_iters = int(getattr(model, "n_iter_", -1))
    converged = n_iters > 0 and n_iters < max_iter
    if not converged:
        warnings.warn(
            f"kuant.ica: FastICA did not converge in {max_iter} "
            f"iterations (tol={tol}); returned mixing/unmixing may be "
            f"unreliable.  [KW-CONV-MAXITER]",
            KuantNumericWarning,
            stacklevel=2,
        )

    return IcaResult(
        sources=sources,
        mixing=mixing,
        unmixing=unmixing,
        mean=model.mean_.copy(),
        n_iters=n_iters,
        converged=converged,
    )


__all__ = ["IcaResult", "ica"]

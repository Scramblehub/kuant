"""Hierarchical Risk Parity (Lopez de Prado 2016).

Three-step allocation:
1. Convert correlation matrix to a distance matrix.
2. Cluster assets hierarchically (single linkage on the distance).
3. Recursively bisect the clustered tree, allocating capital via
   inverse-variance weights between sibling subclusters at each level.

HRP is quasi-diagonal by construction: it never solves a matrix
inversion, making it robust to singular covariance matrices that
break classical mean-variance optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class HrpResult:
    weights: np.ndarray
    order: np.ndarray  # asset indices in the clustered order
    linkage: np.ndarray  # scipy-style Z matrix

    def summary(self) -> str:
        return (
            "=== HrpResult ===\n"
            f"n assets:    {self.weights.size}\n"
            f"weight sum:  {self.weights.sum():.4f}\n"
            f"max weight:  {self.weights.max():.4f}\n"
            f"min weight:  {self.weights.min():.4f}"
        )


def _corr_to_dist(corr: np.ndarray) -> np.ndarray:
    # HRP distance: sqrt((1 - corr) / 2), bounded in [0, 1].
    d = np.sqrt(np.clip((1 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


def _cluster_order(linkage: np.ndarray, n: int) -> np.ndarray:
    """Return leaf order induced by the linkage tree."""
    order = list(linkage[-1, :2].astype(int))
    ordered = []

    def _expand(node):
        if node < n:
            ordered.append(int(node))
            return
        # Internal node: linkage index is node - n.
        idx = int(node - n)
        left = int(linkage[idx, 0])
        right = int(linkage[idx, 1])
        _expand(left)
        _expand(right)

    for node in order:
        _expand(node)
    return np.asarray(ordered, dtype=int)


def _inverse_variance_pair(cov: np.ndarray, group_a, group_b) -> float:
    # Weight fraction on group_a: alpha = 1 - var_a / (var_a + var_b)
    # where each group's "variance" is the ivp-weighted portfolio variance.
    def _ivp_var(idx):
        cvec = cov[np.ix_(idx, idx)]
        diag = np.diag(cvec)
        inv = 1.0 / np.where(diag > 0, diag, 1e-12)
        w = inv / inv.sum()
        return float(w @ cvec @ w)

    va = _ivp_var(group_a)
    vb = _ivp_var(group_b)
    return 1.0 - va / (va + vb) if (va + vb) > 0 else 0.5


def hrp(cov, corr=None) -> HrpResult:
    """Hierarchical risk parity.

    Parameters
    ----------
    cov : 2D array, shape (n, n)
        Asset covariance matrix.
    corr : 2D array, shape (n, n), optional
        Correlation matrix. If None, computed from cov.

    Returns
    -------
    HrpResult

    References
    ----------
    Lopez de Prado 2016, "Building diversified portfolios that
    outperform out-of-sample."
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise KuantShapeError(
            f"kuant.hrp: 'cov' must be a square matrix; got shape " f"{cov.shape}.  [KE-SHAPE-2D]"
        )
    n = cov.shape[0]
    require_positive(n, "n_assets", kernel="hrp", kind="int")
    if corr is None:
        d_ = np.sqrt(np.diag(cov))
        d_ = np.where(d_ > 0, d_, 1e-12)
        corr = cov / np.outer(d_, d_)
    else:
        corr = np.asarray(corr, dtype=np.float64)
        if corr.shape != cov.shape:
            raise KuantShapeError(
                f"kuant.hrp: corr shape {corr.shape} must match cov "
                f"shape {cov.shape}.  [KE-SHAPE-2D]"
            )

    dist = _corr_to_dist(corr)

    try:
        from scipy.cluster.hierarchy import linkage as _linkage

        # Condense the distance matrix for scipy.
        iu = np.triu_indices(n, k=1)
        condensed = dist[iu]
        Z = _linkage(condensed, method="single")
    except ImportError as e:
        raise KuantValueError(
            "kuant.hrp: requires scipy for hierarchical clustering.  "
            "[KE-DEP-MISSING]\n"
            "  -> Fix: pip install scipy"
        ) from e

    order = _cluster_order(Z, n)

    # Recursive bisection.
    weights = np.ones(n, dtype=np.float64)

    def _recurse(idx_list):
        if len(idx_list) <= 1:
            return
        half = len(idx_list) // 2
        left = list(idx_list[:half])
        right = list(idx_list[half:])
        alpha = _inverse_variance_pair(cov, left, right)
        for i in left:
            weights[i] *= alpha
        for j in right:
            weights[j] *= 1.0 - alpha
        _recurse(left)
        _recurse(right)

    _recurse(order.tolist())
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    return HrpResult(weights=weights, order=order, linkage=Z)


__all__ = ["HrpResult", "hrp"]

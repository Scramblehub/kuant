"""PC algorithm skeleton (Spirtes-Glymour-Scheines).

Estimates the undirected causal skeleton of a Bayesian network from
observational data via conditional independence tests. Given variables
V and observed data X in R^{n x |V|}, returns an adjacency matrix
adj[i,j] = 1 if edge i - j survives all conditional independence
tests up to order `max_order`.

Implementation is the SKELETON phase only (Meek-orientation is not
implemented in v0.6 -- callers get a CPDAG-precursor skeleton, which
is what most causal-discovery pipelines actually consume). CI test is
partial-correlation Fisher-Z (standard for Gaussian data). For binary
or ordinal data use a chi-square variant (not in this kernel).

Complexity: O(|V|^(max_order+2)) worst case. Set `max_order` to 2 or 3
for tractable runs on 20+ variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from kuant._validation import require_2d, require_positive, require_range
from kuant.errors import KuantValueError


@dataclass
class PcAlgoResult:
    adj: np.ndarray
    sepsets: dict
    n_ci_tests: int
    max_order_used: int
    alpha: float
    n: int
    p: int

    def summary(self) -> str:
        edges = int(np.sum(self.adj) // 2)
        return (
            "=== PcAlgoResult ===\n"
            f"skeleton edges:       {edges}\n"
            f"CI tests performed:   {self.n_ci_tests}\n"
            f"max order reached:    {self.max_order_used}\n"
            f"alpha:                {self.alpha}\n"
            f"n / p:                {self.n} / {self.p}"
        )


def _partial_corr(data: np.ndarray, i: int, j: int, S: tuple) -> float:
    """Partial correlation of X_i, X_j given X_S via matrix inversion."""
    idx = list(S) + [i, j]
    sub = data[:, idx]
    cov = np.cov(sub, rowvar=False)
    prec = np.linalg.pinv(cov)
    # Last two entries in prec are the (i, j) block after conditioning.
    p = -prec[-2, -1] / np.sqrt(prec[-2, -2] * prec[-1, -1] + 1e-30)
    return float(np.clip(p, -0.999999, 0.999999))


def _fisher_z_pvalue(rho: float, n: int, k: int) -> float:
    """Two-sided p-value for partial correlation via Fisher-Z."""
    if n - k - 3 <= 0:
        return 1.0
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    stat = z * np.sqrt(n - k - 3)
    # Two-sided normal CDF via erfc.
    from math import erfc, sqrt

    p = float(erfc(abs(stat) / sqrt(2.0)))
    return p


def pcalgo(
    data,
    *,
    alpha: float = 0.05,
    max_order: int = 3,
) -> PcAlgoResult:
    """PC algorithm skeleton phase (Spirtes-Glymour-Scheines).

    Parameters
    ----------
    data : 2D array (n, p)
        Continuous observed data, one column per variable.
    alpha : float, default 0.05
        Fisher-Z CI-test significance level. Higher = keeps more edges.
    max_order : int, default 3
        Maximum conditioning set size to test. Bounds runtime.

    Returns
    -------
    PcAlgoResult

    References
    ----------
    Spirtes, Glymour & Scheines 2000 ("Causation, Prediction, and Search");
    Kalisch & Buhlmann 2007 for the Fisher-Z variant used here.
    """
    X = np.asarray(data, dtype=np.float64)
    require_2d(X, "data", kernel="pcalgo")
    n, p = X.shape
    if n < 30:
        raise KuantValueError(
            f"kuant.pcalgo: only {n} rows; need at least 30 for stable "
            f"CI tests.  [KE-VAL-MIN-CLEAN]"
        )
    if p < 2:
        raise KuantValueError(
            f"kuant.pcalgo: need at least 2 variables; got {p}.  " f"[KE-VAL-RANGE]"
        )
    require_range(
        alpha, "alpha", kernel="pcalgo", lo=0.0, hi=1.0, lo_inclusive=False, hi_inclusive=False
    )
    require_positive(max_order, "max_order", kernel="pcalgo", kind="int")

    adj = np.ones((p, p), dtype=np.int8) - np.eye(p, dtype=np.int8)
    sepsets: dict = {}
    n_tests = 0
    reached = 0

    for order in range(0, max_order + 1):
        reached = order
        # Snapshot edges to iterate; deletion happens as we go.
        edges = [(i, j) for i in range(p) for j in range(i + 1, p) if adj[i, j] == 1]
        for i, j in edges:
            if adj[i, j] == 0:
                continue
            neighbors = [
                k for k in range(p) if k != i and k != j and (adj[i, k] == 1 or adj[j, k] == 1)
            ]
            if len(neighbors) < order:
                continue
            for S in combinations(neighbors, order):
                rho = _partial_corr(X, i, j, S)
                pval = _fisher_z_pvalue(rho, n, len(S))
                n_tests += 1
                if pval > alpha:
                    adj[i, j] = 0
                    adj[j, i] = 0
                    sepsets[(i, j)] = tuple(S)
                    sepsets[(j, i)] = tuple(S)
                    break
        # Textbook PC termination (Spirtes-Glymour-Scheines / Kalisch-
        # Buhlmann): stop when no remaining edge has enough neighbors to
        # condition on at the NEXT order. This handles cases where an edge
        # survives order K but is separable at order K+1 with a larger
        # conditioning set (missed by a "no removal at K -> break" rule).
        max_neighbors = 0
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] == 1:
                    cnt = sum(
                        1
                        for k in range(p)
                        if k != i and k != j and (adj[i, k] == 1 or adj[j, k] == 1)
                    )
                    if cnt > max_neighbors:
                        max_neighbors = cnt
        if order + 1 > max_neighbors:
            break

    return PcAlgoResult(
        adj=adj,
        sepsets=sepsets,
        n_ci_tests=int(n_tests),
        max_order_used=int(reached),
        alpha=float(alpha),
        n=int(n),
        p=int(p),
    )


__all__ = ["PcAlgoResult", "pcalgo"]

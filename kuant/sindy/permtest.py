"""Permutation null-hypothesis test — universal utility.

Motivation. Cross-validated R² and Granger F-tests give you a
"probability of chance" number that assumes clean statistical
regularity conditions. Financial data rarely satisfies those. The
gold standard is a **permutation test**: shuffle the target, refit
the signal, and see how often the shuffled data produces a metric
at least as extreme as your real data.

If your real metric is in the tail of the permuted distribution
(p < 0.05), the signal is real. If your real metric is
indistinguishable from a shuffle (p ≈ 0.5), the "signal" is noise
you happened to fit — the model would look "just as good" on
random garbage.

This is the single most important null test for a null-heavy
research pipeline. Ran hundreds of these on our own strategies; without them
we would have shipped garbage signals.

Design: docs/tools/permtest.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantValueError


@dataclass
class PermutationTestResult:
    """Outcome of a permutation null-hypothesis test."""

    real_metric: float
    permuted_metrics: np.ndarray
    p_value: float  # fraction of perm metrics >= real
    n_perms: int
    at_least_as_extreme: int

    def summary(self) -> str:
        return (
            f"=== Permutation test ===\n"
            f"Real metric:       {self.real_metric:.6f}\n"
            f"Perm median:       {np.median(self.permuted_metrics):.6f}\n"
            f"Perm 95%ile:       {np.quantile(self.permuted_metrics, 0.95):.6f}\n"
            f"Perm 99%ile:       {np.quantile(self.permuted_metrics, 0.99):.6f}\n"
            f"At-least-as-extreme: {self.at_least_as_extreme} / {self.n_perms}\n"
            f"p-value:           {self.p_value:.4f}\n"
            f"Signal is real:    {self.p_value < 0.05}"
        )


def permtest(
    real_metric: float,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    n_perms: int = 1000,
    seed: int = 0,
    higher_is_better: bool = True,
) -> PermutationTestResult:
    """Run a permutation null-hypothesis test.

    Parameters
    ----------
    real_metric : float
        The metric value you computed on the un-shuffled data.
    metric_fn : callable
        `metric_fn(x, y_shuffled) -> float`. Computes the same metric
        used to derive `real_metric` but on shuffled targets.
    x : np.ndarray
        Features (shape [n_samples, ...]).
    y : np.ndarray
        Target (shape [n_samples]).
    n_perms : int, default 1000
        Number of permutations to run.
    seed : int, default 0
        Random seed for reproducibility.
    higher_is_better : bool, default True
        If True, p-value = P(permuted >= real). If False, P(permuted <= real).

    Returns
    -------
    PermutationTestResult

    Notes
    -----
    Standard convention: p_value = (# at-least-as-extreme + 1) / (n_perms + 1)
    The +1 correction avoids p=0 for tiny n_perms and is asymptotically
    correct.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(500, 3))
    >>> y = x @ [0.5, 0.0, 0.0] + rng.normal(scale=0.5, size=500)
    >>> def r2_fn(X, y):
    ...     m = LinearRegression().fit(X, y)
    ...     return m.score(X, y)
    >>> real_r2 = r2_fn(x, y)
    >>> result = permtest(real_r2, r2_fn, x, y, n_perms=100)
    >>> result.p_value < 0.05  # signal is real
    True
    """
    require_positive(n_perms, "n_perms", kernel="permtest", kind="int")
    if not np.isfinite(real_metric):
        raise KuantValueError(
            f"kuant.permtest: 'real_metric' must be finite, got "
            f"{real_metric}; comparison to the permuted distribution is "
            f"undefined (all comparisons against NaN are False, yielding a "
            f"falsely-significant p-value of 1/(n_perms+1)).  "
            f"[KE-VAL-FINITE]\n"
            f"  → Fix: compute a finite metric on the unshuffled data "
            f"before calling permtest"
        )

    rng = np.random.default_rng(seed)
    perm_metrics = np.empty(n_perms)
    for i in range(n_perms):
        y_shuffled = rng.permutation(y)
        perm_metrics[i] = metric_fn(x, y_shuffled)

    if higher_is_better:
        at_least_as_extreme = int(np.sum(perm_metrics >= real_metric))
    else:
        at_least_as_extreme = int(np.sum(perm_metrics <= real_metric))

    p_value = (at_least_as_extreme + 1) / (n_perms + 1)

    return PermutationTestResult(
        real_metric=real_metric,
        permuted_metrics=perm_metrics,
        p_value=p_value,
        n_perms=n_perms,
        at_least_as_extreme=at_least_as_extreme,
    )

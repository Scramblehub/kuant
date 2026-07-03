"""Hill estimator for the tail index of a positive series.

    tailindex(x, k_frac=0.10)
        Sort x descending. Take the top k = ⌈k_frac · n⌉ values.
        Return:
            ξ_hat = (1/k) · Σ_{i=1..k}  log(X_(i)) - log(X_(k+1))

For financial losses, apply to |negative returns| to get the left-tail
index. ξ > 0 → Pareto-like heavy tail; ξ = 0 → exponential; ξ < 0 →
bounded (rare in returns).

Design: docs/kernels/stats/tailindex.md.
"""

from __future__ import annotations

import numpy as np


def tailindex(x, k_frac: float = 0.10, min_k: int = 10) -> float:
    """Hill estimator on the top-k order statistics of `x`.

    Parameters
    ----------
    x : 1D array
        Positive values (typically loss magnitudes). NaN and non-positive
        entries are filtered before ranking.
    k_frac : float, default 0.10
        Fraction of the sample used as the tail. `k = max(min_k, ⌈k_frac·n⌉)`.
    min_k : int, default 10
        Absolute floor on the tail size. Tail with fewer values yields NaN.

    Returns
    -------
    float
        Hill estimate ξ_hat. NaN if there aren't enough valid samples.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Pareto with true xi = 0.5
    >>> pareto = (1 - rng.uniform(size=10000)) ** (-0.5)
    >>> abs(tailindex(pareto, k_frac=0.05) - 0.5) < 0.1
    True
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < min_k + 2:
        return float("nan")

    k = max(min_k, int(np.ceil(k_frac * arr.size)))
    if k >= arr.size:
        return float("nan")

    sorted_desc = np.sort(arr)[::-1]
    top = sorted_desc[:k]
    threshold = sorted_desc[k]
    log_ratios = np.log(top) - np.log(threshold)
    return float(np.mean(log_ratios))

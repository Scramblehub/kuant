"""Multiple-hypothesis correction methods.

When you test N strategies and pick the best, the observed
significance is inflated. Standard corrections:

- **Bonferroni**: multiply raw p by N. Simplest, most conservative.
- **Holm**: step-down variant of Bonferroni. Uniformly more powerful.
- **Benjamini-Hochberg**: controls the False Discovery Rate rather
  than the family-wise error rate. Less conservative; standard in
  research settings where a few false positives are tolerable.

Design: docs/kernels/nulltest/mht_correction.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError


_ALLOWED = ("bonferroni", "holm", "bh")


def mht_correction(p_values, method: str = "bh") -> np.ndarray:
    """Apply multiple-hypothesis correction to a set of p-values.

    Parameters
    ----------
    p_values : 1D array or scalar
        Raw one-hypothesis p-values in `[0, 1]`.
    method : {'bonferroni', 'holm', 'bh'}, default 'bh'

    Returns
    -------
    1D np.ndarray or float
        Adjusted p-values, capped at 1.0. Same shape as input.

    Notes
    -----
    - Bonferroni: `p_adj = min(1, N * p)`. Controls family-wise error
      rate at exactly `alpha` under any dependence.
    - Holm: sort ascending, adjust by `(N - k) * p` for the k-th
      smallest (k=0..N-1). Enforce monotonicity by cumulative max.
    - BH: sort ascending, adjust by `N * p / (k + 1)`. Enforce
      monotonicity from the top down. Controls False Discovery Rate.

    Examples
    --------
    >>> import numpy as np
    >>> raw = np.array([0.001, 0.01, 0.04, 0.20, 0.50])
    >>> mht_correction(raw, method='bonferroni').tolist()
    [0.005, 0.05, 0.2, 1.0, 1.0]
    """
    if method not in _ALLOWED:
        raise KuantValueError(
            f"kuant.mht_correction: 'method' must be one of {_ALLOWED}, "
            f"got {method!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED}"
        )
    scalar_input = np.isscalar(p_values)
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if not scalar_input:
        require_1d(p, "p_values", kernel="mht_correction")
    nan_mask = np.isnan(p)
    if bool(nan_mask.any()):
        bad = int(np.argmax(nan_mask))
        raise KuantValueError(
            f"kuant.mht_correction: p-value at index {bad} is NaN; "
            f"adjusted values would be arbitrary because sort order of "
            f"NaN is undefined.  [KE-VAL-NAN-PVALUES]\n"
            f"  → Fix: drop or impute NaN p-values before correction; "
            f"NaN typically comes from a failed t-test on a constant "
            f"series upstream"
        )
    if bool((p < 0).any()) or bool((p > 1).any()):
        bad = int(np.argmax((p < 0) | (p > 1)))
        raise KuantValueError(
            f"kuant.mht_correction: p-value at index {bad} = {p[bad]} is "
            f"outside [0, 1].  [KE-VAL-PROBABILITY]\n"
            f"  → Fix: clip or debug the source of the p-values"
        )

    N = p.size
    if method == "bonferroni":
        adj = np.minimum(p * N, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        sorted_p = p[order]
        scaled = sorted_p * (N - np.arange(N))
        # Enforce monotone non-decrease.
        scaled = np.maximum.accumulate(scaled)
        adj_sorted = np.minimum(scaled, 1.0)
        adj = np.empty_like(p)
        adj[order] = adj_sorted
    else:  # bh
        order = np.argsort(p)
        sorted_p = p[order]
        ranks = np.arange(1, N + 1)
        scaled = sorted_p * N / ranks
        # BH monotone: enforce from top down (walk backward).
        adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
        adj = np.empty_like(p)
        adj[order] = adj_sorted

    return float(adj[0]) if scalar_input else adj


__all__ = ["mht_correction"]

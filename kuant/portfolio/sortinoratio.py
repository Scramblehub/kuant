"""Annualized full-history Sortino ratio.

Sharpe uses the full standard deviation as the risk measure. Sortino
uses only the DOWNSIDE deviation:

    downside_std = sqrt(mean(min(returns - target, 0)^2))
    sortino = (mean - target) * sqrt(ann_factor) / downside_std

The intuition: upside volatility is not risk. A strategy with heavy
right-tail returns should be rewarded, not penalized.

`target` is often 0 (the Minimum Acceptable Return convention) or
the risk-free rate per period.

For a rolling Sortino over a trailing window use
`kuant.stats.rollsortino`. This kernel is the full-history scalar.

Design: docs/kernels/portfolio/sortinoratio.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, warn_kuant, warn_zero_denominator
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class SortinoResult:
    """Full-history annualized Sortino plus its components.

    Attributes
    ----------
    sortino : float
        Annualized Sortino ratio.
    mean_excess : float
        Per-period mean of `returns - target`.
    downside_std : float
        Root-mean-square of the negative excursions
        `min(returns - target, 0)`.
    n : int
        Number of finite return observations used.
    n_below_target : int
        How many observations fell below `target`. Sortino is
        undefined when zero.
    ann_factor : float
    target : float
        Per-period target return used in the calculation.
    """

    sortino: float
    mean_excess: float
    downside_std: float
    n: int
    n_below_target: int
    ann_factor: float
    target: float

    def summary(self) -> str:
        parts = [
            "=== SortinoResult ===",
            f"annualized Sortino:  {self.sortino:+.4f}",
            f"per-period mean-tgt: {self.mean_excess:+.6f}",
            f"downside std:        {self.downside_std:.6f}",
            f"n observations:      {self.n}",
            f"n below target:      {self.n_below_target}",
            f"ann_factor:          {self.ann_factor:g}",
            f"target per period:   {self.target:g}",
        ]
        return "\n".join(parts)


def sortinoratio(
    returns,
    ann_factor: float = 252,
    target: float = 0.0,
) -> SortinoResult:
    """Annualized full-history Sortino ratio.

    Parameters
    ----------
    returns : 1D array
        Periodic returns. NaN is dropped.
    ann_factor : float, default 252
        Same convention as `sharperatio`.
    target : float, default 0.0
        Per-period target return (Minimum Acceptable Return).
        Downside deviation is computed from
        `min(returns - target, 0)`.

    Returns
    -------
    SortinoResult

    Warnings
    --------
    - `KuantNumericWarning` (`KW-SORTINO-SMALL-SAMPLE`) if fewer than
      30 finite observations. Sortino has fatter sampling noise than
      Sharpe because it uses only the downside subsample.
    - `KuantNumericWarning` (`KW-SORTINO-NO-DOWNSIDE`) if zero
      observations fell below `target`. Sortino is undefined
      (division by zero); returned as +∞ if mean_excess > 0, else 0.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> daily = rng.normal(0.001, 0.01, 2520)
    >>> r = sortinoratio(daily, ann_factor=252)
    >>> r.sortino > 0                    # positive drift → positive Sortino
    True
    """
    require_positive(ann_factor, "ann_factor", kernel="sortinoratio")

    arr = np.asarray(returns, dtype=np.float64)
    require_1d(arr, "returns", kernel="sortinoratio")
    finite = arr[np.isfinite(arr)]
    n = finite.size

    if n == 0:
        raise KuantValueError(
            "kuant.sortinoratio: 'returns' has no finite values.  "
            "[KE-VAL-FINITE]\n"
            "  → Fix: provide at least one finite return"
        )
    if n < 30:
        warn_kuant(
            kernel="sortinoratio",
            code="KW-SORTINO-SMALL-SAMPLE",
            what=(
                f"only {n} finite observations; Sortino has fatter "
                f"sampling noise than Sharpe below n=30"
            ),
            fix=(
                "collect more data, or interpret the result as a rough "
                "estimate; downside subsample size drives the "
                "uncertainty"
            ),
            category=KuantNumericWarning,
        )

    excess = finite - float(target)
    mean_excess = float(np.mean(excess))
    downside = np.minimum(excess, 0.0)
    n_below = int((downside < 0).sum())
    # Sortino divides by RMS of downside excursions (over the full sample,
    # NOT just the downside subsample — the standard convention).
    downside_std = float(np.sqrt(np.mean(downside * downside)))

    if n_below == 0:
        warn_kuant(
            kernel="sortinoratio",
            code="KW-SORTINO-NO-DOWNSIDE",
            what=(
                f"no observations fell below target ({target:g}); " f"Sortino denominator is zero"
            ),
            fix=(
                "either the strategy really never lost (rare and "
                "worth verifying) or the target is set too low; "
                "sortino returned as ±inf or 0 by convention"
            ),
            category=KuantNumericWarning,
        )
        if mean_excess > 0:
            sortino = float("inf")
        elif mean_excess < 0:
            sortino = float("-inf")
        else:
            sortino = 0.0
    elif downside_std < 1e-15:
        # Downside excursions exist but are tiny relative to mean.
        warn_zero_denominator("downside_std", "sortinoratio", code="KW-SORTINO-TINY-DOWNSIDE")
        sortino = 0.0
    else:
        sortino = mean_excess * np.sqrt(float(ann_factor)) / downside_std

    return SortinoResult(
        sortino=float(sortino),
        mean_excess=mean_excess,
        downside_std=downside_std,
        n=n,
        n_below_target=n_below,
        ann_factor=float(ann_factor),
        target=float(target),
    )


__all__ = ["sortinoratio", "SortinoResult"]

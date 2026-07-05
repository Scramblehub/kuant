"""Annualized full-history Sharpe ratio.

For a periodic return series and an annualization factor:

    sharpe = (mean(returns) - rf_per_period) * sqrt(ann_factor) / std(returns)

Common annualization factors:

    ann_factor = 252     daily returns
    ann_factor = 52      weekly
    ann_factor = 12      monthly
    ann_factor = 1       already annual

For a rolling Sharpe over a trailing window use
`kuant.stats.rollsharpe`. This kernel is the full-history scalar.

Design: docs/kernels/portfolio/sharperatio.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, warn_kuant
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class SharpeResult:
    """Full-history annualized Sharpe plus its components.

    Attributes
    ----------
    sharpe : float
        Annualized Sharpe ratio.
    mean : float
        Per-period mean return (excess of rf if supplied).
    std : float
        Per-period std of returns (sample std, ddof=1).
    n : int
        Number of finite return observations used.
    ann_factor : float
    rf : float
        The per-period risk-free rate that was subtracted.
    """

    sharpe: float
    mean: float
    std: float
    n: int
    ann_factor: float
    rf: float

    def summary(self) -> str:
        parts = [
            "=== SharpeResult ===",
            f"annualized Sharpe:   {self.sharpe:+.4f}",
            f"per-period mean:     {self.mean:+.6f}",
            f"per-period std:      {self.std:.6f}",
            f"n observations:      {self.n}",
            f"ann_factor:          {self.ann_factor:g}",
            f"rf per period:       {self.rf:g}",
        ]
        return "\n".join(parts)


def sharperatio(
    returns,
    ann_factor: float = 252,
    rf: float = 0.0,
) -> SharpeResult:
    """Annualized full-history Sharpe.

    Parameters
    ----------
    returns : 1D array
        Periodic returns. NaN is dropped before computation.
    ann_factor : float, default 252
        Multiplier applied inside the Sharpe formula:
        `sharpe = excess_mean * sqrt(ann_factor) / excess_std`.
        Use 252 for daily, 52 for weekly, 12 for monthly, 1 for annual.
    rf : float, default 0.0
        Risk-free rate PER PERIOD (not annual). Subtract from every
        return before computing mean and std. If your rf is annual,
        divide by `ann_factor` before passing in.

    Returns
    -------
    SharpeResult

    Warnings
    --------
    `KuantNumericWarning` (`KW-SHARPE-SMALL-SAMPLE`) if the number of
    finite observations is below 30. The Sharpe estimate is dominated
    by sampling noise at that scale.

    Notes
    -----
    - Zero std (constant returns) returns Sharpe = 0 as a convention.
    - NaN is silently dropped. Non-finite `returns` never contribute.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> daily = rng.normal(0.001, 0.01, 2520)  # 10y daily
    >>> r = sharperatio(daily, ann_factor=252)
    >>> r.n
    2520
    """
    require_positive(ann_factor, "ann_factor", kernel="sharperatio")

    arr = np.asarray(returns, dtype=np.float64)
    require_1d(arr, "returns", kernel="sharperatio")
    finite = arr[np.isfinite(arr)]
    n = finite.size

    if n == 0:
        raise KuantValueError(
            "kuant.sharperatio: 'returns' has no finite values.  "
            "[KE-VAL-FINITE]\n"
            "  → Fix: provide at least one finite return"
        )
    if n < 30:
        warn_kuant(
            kernel="sharperatio",
            code="KW-SHARPE-SMALL-SAMPLE",
            what=(
                f"only {n} finite observations; Sharpe estimate is "
                f"dominated by sampling noise below n=30"
            ),
            fix=(
                "collect more data, or interpret the result as a rough "
                "estimate; the standard error on Sharpe scales like "
                "1/sqrt(n) so the uncertainty is large at low n"
            ),
            category=KuantNumericWarning,
        )

    excess = finite - float(rf)
    mean = float(np.mean(excess))
    std = float(np.std(excess, ddof=1)) if n > 1 else 0.0

    # Effectively-constant returns produce a std at the level of FP noise
    # rather than exactly 0. Guard against dividing by that noise since it
    # would generate a nonsense huge Sharpe.
    if std < 1e-15:
        sharpe = 0.0
    else:
        sharpe = mean * np.sqrt(float(ann_factor)) / std

    return SharpeResult(
        sharpe=float(sharpe),
        mean=mean,
        std=std,
        n=n,
        ann_factor=float(ann_factor),
        rf=float(rf),
    )


__all__ = ["sharperatio", "SharpeResult"]

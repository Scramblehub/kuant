"""Cross-sectional factor scoring: IC, quantile spreads, rank autocorrelation.

Panel inputs are shape `(T, N)`: rows are dates, columns are names.
At each date, a factor score across the N names is compared against
that date's forward returns across the same N names.

Distinct from `icdecay`, which is time-series-univariate (signal[t] vs
`sum(ret[t+1..t+h])`). Here everything is CROSS-SECTIONAL: rank the
names at each date, and see how the top-quintile names' next-period
returns compare to the bottom-quintile.

Design: docs/kernels/signals/factorscoring.md.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_2d,
    require_dep,
    require_positive,
    require_range,
)
from kuant.errors import KuantShapeError, KuantValueError

_ALLOWED_IC_METHODS = ("spearman", "pearson", "kendall")


# ---------- factor_ic ----------------------------------------------------


@dataclass
class FactorICResult:
    """Per-period cross-sectional Information Coefficient.

    Attributes
    ----------
    ic : 1D np.ndarray, length T
        IC at each period. NaN where fewer than 3 clean pairs.
    mean : float
        Mean IC across periods with a finite value.
    std : float
        Std of the IC series (sample, ddof=1).
    ir : float
        Information Ratio = mean / std * sqrt(T_finite). The
        annualization convention used depends on the input's cadence;
        we report the raw sqrt(T) version.
    t_stat : float
        `mean / (std / sqrt(T_finite))`. |t| > 2 rule of thumb.
    n_periods : int
        Number of periods with a finite IC.
    method : str
    """

    ic: np.ndarray
    mean: float
    std: float
    ir: float
    t_stat: float
    n_periods: int
    method: str

    def summary(self) -> str:
        parts = [
            "=== FactorICResult ===",
            f"method:          {self.method}",
            f"n_periods:       {self.n_periods}",
            f"mean IC:         {self.mean:+.4f}",
            f"IC std:          {self.std:.4f}",
            f"IR (mean/std):   {self.ir:+.4f}",
            f"t-stat:          {self.t_stat:+.4f}",
        ]
        return "\n".join(parts)


def factor_ic(factor, forward_returns, method: str = "spearman") -> FactorICResult:
    """Per-period cross-sectional IC of a factor vs forward returns.

    Parameters
    ----------
    factor : 2D array of shape (T, N)
        Factor scores. NaN cells are excluded from the row correlation.
    forward_returns : 2D array of shape (T, N)
        Next-period returns aligned to `factor`.
    method : {'spearman', 'pearson', 'kendall'}, default 'spearman'
        Rank-order Spearman is the standard factor-research choice.

    Returns
    -------
    FactorICResult

    Notes
    -----
    Requires `scipy.stats` for the correlation. Rows with fewer than 3
    finite `(factor, return)` pairs yield NaN in `ic` (no correlation
    computable).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T, N = 500, 50
    >>> f = rng.standard_normal((T, N))
    >>> r = 0.05 * f + rng.standard_normal((T, N))    # 5% IC by construction
    >>> res = factor_ic(f, r)
    >>> res.mean > 0.02
    True
    """
    if method not in _ALLOWED_IC_METHODS:
        raise KuantValueError(
            f"kuant.factor_ic: 'method' must be one of "
            f"{_ALLOWED_IC_METHODS}, got {method!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_IC_METHODS}"
        )
    try:
        from scipy.stats import kendalltau, pearsonr, spearmanr
    except ImportError as e:
        require_dep(
            "scipy",
            kernel="factor_ic",
            install="pip install scipy",
            cause=e,
        )

    F = np.asarray(factor, dtype=np.float64)
    R = np.asarray(forward_returns, dtype=np.float64)
    require_2d(F, "factor", kernel="factor_ic")
    require_2d(R, "forward_returns", kernel="factor_ic")
    if F.shape != R.shape:
        raise KuantShapeError(
            f"kuant.factor_ic: 'factor' and 'forward_returns' must share "
            f"shape, got {F.shape} vs {R.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: align both to a common (T, N) panel"
        )

    T = F.shape[0]
    ic_arr = np.full(T, np.nan)
    with warnings.catch_warnings():
        # scipy may warn on degenerate rows; we handle NaN outputs cleanly.
        warnings.simplefilter("ignore")
        for t in range(T):
            f = F[t]
            r = R[t]
            mask = np.isfinite(f) & np.isfinite(r)
            if int(mask.sum()) < 3:
                continue
            if method == "spearman":
                rho, _ = spearmanr(f[mask], r[mask])
            elif method == "pearson":
                rho, _ = pearsonr(f[mask], r[mask])
            else:
                rho, _ = kendalltau(f[mask], r[mask])
            if np.isfinite(rho):
                ic_arr[t] = float(rho)

    finite = ic_arr[np.isfinite(ic_arr)]
    n = finite.size
    mean = float(finite.mean()) if n > 0 else float("nan")
    std = float(finite.std(ddof=1)) if n > 1 else 0.0
    ir = mean / std * np.sqrt(n) if std > 0 and n > 0 else 0.0
    t_stat = mean / (std / np.sqrt(n)) if std > 0 and n > 0 else 0.0

    return FactorICResult(
        ic=ic_arr,
        mean=mean,
        std=std,
        ir=float(ir),
        t_stat=float(t_stat),
        n_periods=n,
        method=method,
    )


# ---------- factor_rank_autocorr ---------------------------------------


@dataclass
class RankAutocorrResult:
    """Cross-sectional rank autocorrelation of a factor across time.

    High rank autocorrelation means the factor's cross-sectional
    ranking is stable across periods. Low or negative values mean the
    ranking flips (high turnover implied).

    Attributes
    ----------
    autocorr : 1D np.ndarray, length T
        Rank autocorrelation at each period vs `lag` periods prior.
        NaN in the first `lag` positions.
    mean : float
    lag : int
    n_periods : int
    """

    autocorr: np.ndarray
    mean: float
    lag: int
    n_periods: int

    def summary(self) -> str:
        parts = [
            "=== RankAutocorrResult ===",
            f"lag:               {self.lag}",
            f"n_periods:         {self.n_periods}",
            f"mean autocorr:     {self.mean:+.4f}",
        ]
        return "\n".join(parts)


def factor_rank_autocorr(factor, lag: int = 1) -> RankAutocorrResult:
    """Cross-sectional Spearman rank autocorrelation across time.

    For each `t >= lag`, compute Spearman correlation between the
    factor ranking at date `t` and the ranking at date `t - lag`,
    across the N names.

    Parameters
    ----------
    factor : 2D array of shape (T, N)
    lag : int, default 1
        Number of periods between the two rankings.

    Returns
    -------
    RankAutocorrResult

    Notes
    -----
    Interpretation:
    - Near +1: factor ranking barely changes period-over-period (slow
      signal, low turnover implied).
    - Near 0: rankings are independent (high turnover implied).
    - Negative: ranking systematically flips.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Persistent factor: today's score is 0.9 * yesterday's + noise.
    >>> T, N = 200, 50
    >>> f = np.zeros((T, N))
    >>> f[0] = rng.standard_normal(N)
    >>> for t in range(1, T):
    ...     f[t] = 0.9 * f[t-1] + 0.1 * rng.standard_normal(N)
    >>> res = factor_rank_autocorr(f)
    >>> res.mean > 0.8                                      # highly persistent
    True
    """
    require_positive(lag, "lag", kernel="factor_rank_autocorr", kind="int")
    try:
        from scipy.stats import spearmanr
    except ImportError as e:
        require_dep(
            "scipy",
            kernel="factor_rank_autocorr",
            install="pip install scipy",
            cause=e,
        )

    F = np.asarray(factor, dtype=np.float64)
    require_2d(F, "factor", kernel="factor_rank_autocorr")

    T = F.shape[0]
    if lag >= T:
        raise KuantValueError(
            f"kuant.factor_rank_autocorr: 'lag' {lag} >= T {T}; need at "
            f"least one (t, t-lag) pair.  [KE-VAL-RANGE]\n"
            f"  → Fix: pass more data or a shorter lag"
        )

    autocorr = np.full(T, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(lag, T):
            a = F[t]
            b = F[t - lag]
            mask = np.isfinite(a) & np.isfinite(b)
            if int(mask.sum()) < 3:
                continue
            rho, _ = spearmanr(a[mask], b[mask])
            if np.isfinite(rho):
                autocorr[t] = float(rho)

    finite = autocorr[np.isfinite(autocorr)]
    mean = float(finite.mean()) if finite.size > 0 else float("nan")
    return RankAutocorrResult(
        autocorr=autocorr,
        mean=mean,
        lag=int(lag),
        n_periods=int(finite.size),
    )


# ---------- quantile bucketing helpers ---------------------------------


def _quantile_bins_per_row(F: np.ndarray, n_quantiles: int) -> np.ndarray:
    """Assign each cell a bin index in [0, n_quantiles); NaN cells → -1."""
    T, N = F.shape
    bins = np.full((T, N), -1, dtype=np.int64)
    edges = np.linspace(0, 1, n_quantiles + 1)
    for t in range(T):
        row = F[t]
        mask = np.isfinite(row)
        if int(mask.sum()) < n_quantiles:
            continue
        vals = row[mask]
        qs = np.quantile(vals, edges[1:-1])
        # np.digitize on the row's finite values against the interior
        # cut points gives an index in [0, n_quantiles).
        idx = np.digitize(vals, qs)
        bins[t, mask] = idx
    return bins


# ---------- mean_return_by_quantile ------------------------------------


@dataclass
class QuantileReturnsResult:
    """Per-period, per-quantile mean forward return.

    Attributes
    ----------
    mean_by_quantile : 2D np.ndarray, shape (T, n_quantiles)
        Row t, col q = mean forward return in that quantile bucket at
        period t.
    total_by_quantile : 1D np.ndarray, length n_quantiles
        Sum across time for each bucket.
    n_quantiles : int
    """

    mean_by_quantile: np.ndarray
    total_by_quantile: np.ndarray
    n_quantiles: int

    def summary(self) -> str:
        parts = ["=== QuantileReturnsResult ===", f"n_quantiles:  {self.n_quantiles}"]
        parts.append("mean total return per bucket (top → bottom):")
        for i in range(self.n_quantiles - 1, -1, -1):
            parts.append(f"  Q{i + 1}:  {self.total_by_quantile[i]:+.6f}")
        return "\n".join(parts)


def mean_return_by_quantile(
    factor,
    forward_returns,
    n_quantiles: int = 5,
) -> QuantileReturnsResult:
    """Bucket each row's names by factor quantile; report per-bucket mean return.

    Parameters
    ----------
    factor : 2D (T, N)
    forward_returns : 2D (T, N)
    n_quantiles : int, default 5

    Returns
    -------
    QuantileReturnsResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = rng.standard_normal((500, 50))
    >>> r = 0.05 * f + rng.standard_normal((500, 50))
    >>> res = mean_return_by_quantile(f, r, n_quantiles=5)
    >>> # Top quantile should out-return bottom.
    >>> res.total_by_quantile[-1] > res.total_by_quantile[0]
    True
    """
    require_range(n_quantiles, "n_quantiles", kernel="mean_return_by_quantile", lo=2, hi=100)
    F = np.asarray(factor, dtype=np.float64)
    R = np.asarray(forward_returns, dtype=np.float64)
    require_2d(F, "factor", kernel="mean_return_by_quantile")
    require_2d(R, "forward_returns", kernel="mean_return_by_quantile")
    if F.shape != R.shape:
        raise KuantShapeError(
            f"kuant.mean_return_by_quantile: shape mismatch {F.shape} vs "
            f"{R.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: align both to a common (T, N) panel"
        )

    T = F.shape[0]
    bins = _quantile_bins_per_row(F, int(n_quantiles))
    mean_by_q = np.full((T, int(n_quantiles)), np.nan)
    for t in range(T):
        for q in range(int(n_quantiles)):
            mask = (bins[t] == q) & np.isfinite(R[t])
            if int(mask.sum()) > 0:
                mean_by_q[t, q] = float(R[t][mask].mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_by_q = np.nansum(mean_by_q, axis=0)

    return QuantileReturnsResult(
        mean_by_quantile=mean_by_q,
        total_by_quantile=total_by_q,
        n_quantiles=int(n_quantiles),
    )


# ---------- quantile_spread ---------------------------------------------


@dataclass
class QuantileSpreadResult:
    """Top-minus-bottom quantile spread series plus scalar summary.

    Attributes
    ----------
    spread : 1D np.ndarray, length T
        Top-quantile mean return minus bottom-quantile mean return
        at each period.
    mean : float
    std : float
    t_stat : float
        `mean / (std / sqrt(n_periods))`.
    n_periods : int
    n_quantiles : int
    """

    spread: np.ndarray
    mean: float
    std: float
    t_stat: float
    n_periods: int
    n_quantiles: int

    def summary(self) -> str:
        parts = [
            "=== QuantileSpreadResult ===",
            f"n_quantiles:      {self.n_quantiles}",
            f"n_periods:        {self.n_periods}",
            f"mean spread:      {self.mean:+.6f}",
            f"std spread:       {self.std:.6f}",
            f"t-stat:           {self.t_stat:+.4f}",
        ]
        return "\n".join(parts)


def quantile_spread(
    factor,
    forward_returns,
    n_quantiles: int = 5,
) -> QuantileSpreadResult:
    """Top-quantile minus bottom-quantile mean forward-return spread series.

    Returns a `QuantileSpreadResult` with the per-period spread, its
    mean, std, and t-statistic against zero.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = rng.standard_normal((500, 50))
    >>> r = 0.05 * f + rng.standard_normal((500, 50))
    >>> res = quantile_spread(f, r, n_quantiles=5)
    >>> res.mean > 0    # top out-earns bottom on average
    True
    >>> abs(res.t_stat) > 2                                  # significant
    True
    """
    qr = mean_return_by_quantile(factor, forward_returns, n_quantiles=n_quantiles)
    Nq = qr.n_quantiles
    spread = qr.mean_by_quantile[:, Nq - 1] - qr.mean_by_quantile[:, 0]
    finite = spread[np.isfinite(spread)]
    n = finite.size
    mean = float(finite.mean()) if n > 0 else float("nan")
    std = float(finite.std(ddof=1)) if n > 1 else 0.0
    t_stat = mean / (std / np.sqrt(n)) if std > 0 and n > 0 else 0.0
    return QuantileSpreadResult(
        spread=spread,
        mean=mean,
        std=std,
        t_stat=float(t_stat),
        n_periods=n,
        n_quantiles=Nq,
    )


# ---------- quantile_turnover ------------------------------------------


@dataclass
class QuantileTurnoverResult:
    """Per-period turnover fraction inside the top and bottom buckets.

    Attributes
    ----------
    top_turnover : 1D np.ndarray
        Fraction of names in the top quantile at period t that were
        NOT in the top quantile at period t-1. NaN in position 0.
    bottom_turnover : 1D np.ndarray
        Same for the bottom quantile.
    top_mean : float
    bottom_mean : float
    n_periods : int
    n_quantiles : int
    """

    top_turnover: np.ndarray
    bottom_turnover: np.ndarray
    top_mean: float
    bottom_mean: float
    n_periods: int
    n_quantiles: int

    def summary(self) -> str:
        parts = [
            "=== QuantileTurnoverResult ===",
            f"n_quantiles:       {self.n_quantiles}",
            f"n_periods:         {self.n_periods}",
            f"mean top-Q churn:  {self.top_mean:.2%}",
            f"mean bot-Q churn:  {self.bottom_mean:.2%}",
        ]
        return "\n".join(parts)


def quantile_turnover(factor, n_quantiles: int = 5) -> QuantileTurnoverResult:
    """Fraction of names that enter/leave the top and bottom buckets.

    Parameters
    ----------
    factor : 2D (T, N)
    n_quantiles : int, default 5

    Returns
    -------
    QuantileTurnoverResult

    Notes
    -----
    Interpretation:
    - `top_turnover[t] = |top_Q(t) - top_Q(t-1)| / |top_Q(t-1)|`.
      That's the fraction of yesterday's top-bucket names that DROPPED
      OUT of today's top bucket. High turnover → transaction-cost drag
      on any strategy going long the top bucket.
    """
    require_range(n_quantiles, "n_quantiles", kernel="quantile_turnover", lo=2, hi=100)
    F = np.asarray(factor, dtype=np.float64)
    require_2d(F, "factor", kernel="quantile_turnover")

    T = F.shape[0]
    bins = _quantile_bins_per_row(F, int(n_quantiles))

    top_turn = np.full(T, np.nan)
    bot_turn = np.full(T, np.nan)
    for t in range(1, T):
        top_now = set(np.where(bins[t] == int(n_quantiles) - 1)[0])
        top_prev = set(np.where(bins[t - 1] == int(n_quantiles) - 1)[0])
        if top_prev:
            top_turn[t] = len(top_prev - top_now) / len(top_prev)
        bot_now = set(np.where(bins[t] == 0)[0])
        bot_prev = set(np.where(bins[t - 1] == 0)[0])
        if bot_prev:
            bot_turn[t] = len(bot_prev - bot_now) / len(bot_prev)

    top_finite = top_turn[np.isfinite(top_turn)]
    bot_finite = bot_turn[np.isfinite(bot_turn)]
    return QuantileTurnoverResult(
        top_turnover=top_turn,
        bottom_turnover=bot_turn,
        top_mean=float(top_finite.mean()) if top_finite.size else float("nan"),
        bottom_mean=float(bot_finite.mean()) if bot_finite.size else float("nan"),
        n_periods=int(max(top_finite.size, bot_finite.size)),
        n_quantiles=int(n_quantiles),
    )


__all__ = [
    "FactorICResult",
    "QuantileReturnsResult",
    "QuantileSpreadResult",
    "QuantileTurnoverResult",
    "RankAutocorrResult",
    "factor_ic",
    "factor_rank_autocorr",
    "mean_return_by_quantile",
    "quantile_spread",
    "quantile_turnover",
]

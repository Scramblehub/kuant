"""Portfolio risk-adjusted return metrics beyond Sharpe/Sortino.

Ships the tearsheet-standard metrics that empyrical and quantstats
made canonical, with kuant's usual result-dataclass + parquet-first
output pattern.

Kernels:

- `omega`: probability-weighted gain/loss ratio above a threshold.
- `ulcer_index`: RMS drawdown depth (pain-path proxy).
- `kelly`: log-optimal sizing fraction.
- `up_capture` / `down_capture`: benchmark-conditional decomposition.
- `probabilistic_sharpe`: Bailey-Lopez de Prado confidence-adjusted Sharpe.
- `deflated_sharpe`: same, adjusting for the number of trials tested.
- `drawdown_table`: per-episode peak/trough/recovery details for the
  top-N drawdowns in an equity curve.

Design: docs/kernels/portfolio/riskmetrics.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_dep,
    require_equal_length,
    require_positive,
    require_range,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantValueError


# ---------- omega ratio -------------------------------------------------


def omega(returns, threshold: float = 0.0) -> float:
    """Omega ratio: E[max(r - t, 0)] / E[max(t - r, 0)] over finite returns.

    The Sharpe alternative for non-Gaussian return distributions.
    An Omega of 1.0 means gains above threshold equal losses below;
    values > 1.0 favor the strategy.

    Parameters
    ----------
    returns : 1D array
        Periodic returns. NaN dropped.
    threshold : float, default 0.0
        The minimum acceptable return per period.

    Returns
    -------
    float
        Omega. Returns `+inf` if there are no downside excursions
        (all returns >= threshold and at least one is strictly above),
        `0.0` if all downside and no upside, `1.0` if both are zero.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.normal(0.001, 0.01, 2520)
    >>> omega(r) > 1.0                                       # positive drift
    True
    """
    arr = np.asarray(returns, dtype=np.float64)
    require_1d(arr, "returns", kernel="omega")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise KuantValueError(
            "kuant.omega: 'returns' has no finite values.  [KE-VAL-FINITE]\n"
            "  → Fix: provide at least one finite return"
        )
    excess = finite - float(threshold)
    upside = np.maximum(excess, 0.0).sum()
    downside = -np.minimum(excess, 0.0).sum()
    if downside == 0.0:
        return float("inf") if upside > 0 else 1.0
    return float(upside / downside)


# ---------- ulcer index -------------------------------------------------


@dataclass
class UlcerResult:
    """Ulcer Index summary.

    Attributes
    ----------
    ulcer_index : float
        RMS of the drawdown series (as a positive number).
    upi : float
        Ulcer Performance Index = mean_return / ulcer_index. Higher is
        better.
    n : int
        Number of bars used.
    """

    ulcer_index: float
    upi: float
    n: int

    def summary(self) -> str:
        return (
            "=== UlcerResult ===\n"
            f"ulcer index:   {self.ulcer_index:.6f}\n"
            f"UPI:           {self.upi:+.4f}\n"
            f"n:             {self.n}"
        )


def ulcer_index(equity) -> UlcerResult:
    """RMS drawdown depth on an equity curve.

    Parameters
    ----------
    equity : 1D array
        Strictly positive equity values (see `drawdown` for the same
        input contract).

    Returns
    -------
    UlcerResult

    Notes
    -----
    The Ulcer Index captures the PAIN PATH: how deep and how long
    drawdowns are, in aggregate. Max drawdown only reports the worst;
    Ulcer weights every drawdown. UPI (Ulcer Performance Index) is
    the sibling of Sharpe using Ulcer instead of std as the risk
    denominator.
    """
    arr = np.asarray(equity, dtype=np.float64)
    require_1d(arr, "equity", kernel="ulcer_index")
    finite = np.isfinite(arr)
    if not bool(finite.any()) or bool((arr[finite] <= 0).any()):
        raise KuantValueError(
            "kuant.ulcer_index: 'equity' must be strictly positive.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: convert returns to equity via np.cumprod(1 + returns)"
        )
    running_max = np.maximum.accumulate(np.where(finite, arr, -np.inf))
    running_max = np.where(running_max == -np.inf, np.nan, running_max)
    dd = arr / running_max - 1.0  # <= 0
    dd_pct = np.abs(dd) * 100.0  # convention: percent
    dd_clean = dd_pct[np.isfinite(dd_pct)]
    if dd_clean.size == 0:
        return UlcerResult(ulcer_index=float("nan"), upi=float("nan"), n=0)
    ui = float(np.sqrt(np.mean(dd_clean * dd_clean)))
    # UPI uses the mean per-bar return of the equity curve.
    logret = np.diff(np.log(arr[finite]))
    mean_ret = float(logret.mean()) * 100.0 if logret.size > 0 else 0.0
    upi = mean_ret / ui if ui > 0 else 0.0
    return UlcerResult(ulcer_index=ui, upi=upi, n=int(dd_clean.size))


# ---------- Kelly criterion --------------------------------------------


def kelly(returns, cap: float = 1.0) -> float:
    """Log-optimal sizing fraction: mean(r) / var(r).

    Parameters
    ----------
    returns : 1D array
        Periodic returns. NaN dropped.
    cap : float, default 1.0
        Upper bound on the returned fraction. Setting < 1 implements
        the "half-Kelly" style safety margin. Must be in (0, 1].

    Returns
    -------
    float
        Kelly fraction clipped to `[0, cap]`. Zero when the mean is
        non-positive.

    Notes
    -----
    The formula `f* = μ / σ²` is the continuous-time / small-return
    approximation of the Kelly criterion. It's what practitioners
    actually use; the exact discrete Kelly requires a per-outcome
    payoff table.
    """
    arr = np.asarray(returns, dtype=np.float64)
    require_1d(arr, "returns", kernel="kelly")
    require_range(cap, "cap", kernel="kelly", lo=0.0, hi=1.0, lo_inclusive=False)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        raise KuantValueError(
            "kuant.kelly: need at least 2 finite returns.  [KE-VAL-RANGE]\n"
            "  → Fix: provide more data"
        )
    mu = float(finite.mean())
    var = float(finite.var(ddof=1))
    if var <= 1e-15:
        return 0.0
    f = mu / var
    if f <= 0:
        return 0.0
    return float(min(f, cap))


# ---------- up / down capture ------------------------------------------


@dataclass
class CaptureResult:
    up_capture: float
    down_capture: float
    up_down: float  # up / down
    n_up: int
    n_down: int

    def summary(self) -> str:
        return (
            "=== CaptureResult ===\n"
            f"up capture:     {self.up_capture:+.4f}\n"
            f"down capture:   {self.down_capture:+.4f}\n"
            f"up/down ratio:  {self.up_down:+.4f}\n"
            f"n up periods:   {self.n_up}\n"
            f"n down periods: {self.n_down}"
        )


def up_capture(returns, benchmark) -> float:
    """Mean return / mean benchmark return over benchmark-up periods."""
    r, b = _align_finite(returns, benchmark, kernel="up_capture")
    up_mask = b > 0
    if int(up_mask.sum()) == 0:
        return 0.0
    return float(r[up_mask].mean() / b[up_mask].mean()) if b[up_mask].mean() != 0 else 0.0


def down_capture(returns, benchmark) -> float:
    """Mean return / mean benchmark return over benchmark-down periods."""
    r, b = _align_finite(returns, benchmark, kernel="down_capture")
    down_mask = b < 0
    if int(down_mask.sum()) == 0:
        return 0.0
    return float(r[down_mask].mean() / b[down_mask].mean()) if b[down_mask].mean() != 0 else 0.0


def _align_finite(a, b, *, kernel: str):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    require_1d(a_arr, "returns", kernel=kernel)
    require_1d(b_arr, "benchmark", kernel=kernel)
    require_equal_length(a_arr, "returns", b_arr, "benchmark", kernel=kernel)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    return a_arr[mask], b_arr[mask]


# ---------- probabilistic + deflated Sharpe ----------------------------


def probabilistic_sharpe(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """Bailey-Lopez de Prado Probabilistic Sharpe Ratio.

    Given an observed Sharpe and its return-series moments, compute
    the probability that the TRUE Sharpe exceeds a benchmark. Named
    "probabilistic" because the output is a probability in [0, 1].

    Parameters
    ----------
    sharpe : float
        Observed annualized Sharpe ratio.
    n : int
        Number of return observations (not the annualization factor).
    skew : float, default 0.0
        Return-series skewness.
    kurt : float, default 3.0
        Return-series kurtosis (raw, NOT excess; Gaussian is 3.0).
    sharpe_benchmark : float, default 0.0
        The Sharpe threshold to beat.

    Returns
    -------
    float
        Probability in `[0, 1]`.

    References
    ----------
    Bailey & Lopez de Prado 2012, "The Sharpe Ratio Efficient Frontier".

    Examples
    --------
    >>> # Sharpe 1.0 on 252 observations vs benchmark 0.0
    >>> probabilistic_sharpe(1.0, 252) > 0.9
    True
    """
    try:
        from scipy.stats import norm
    except ImportError as e:
        require_dep("scipy", kernel="probabilistic_sharpe", install="pip install scipy", cause=e)
    require_positive(n, "n", kernel="probabilistic_sharpe", kind="int")
    if n < 30:
        warn_kuant(
            kernel="probabilistic_sharpe",
            code="KW-PSR-SMALL-SAMPLE",
            what=f"n={n} below 30; PSR CDF approximation degrades on tiny samples",
            fix="collect more return observations",
            category=KuantNumericWarning,
        )
    numer = (sharpe - float(sharpe_benchmark)) * np.sqrt(n - 1)
    denom = np.sqrt(1 - float(skew) * sharpe + ((float(kurt) - 1) / 4.0) * sharpe * sharpe)
    if denom <= 0:
        return 0.0 if numer < 0 else 1.0
    return float(norm.cdf(numer / denom))


def deflated_sharpe(
    sharpe: float,
    n: int,
    n_trials: int,
    variance_of_sharpes: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Bailey-Lopez de Prado Deflated Sharpe Ratio.

    Adjusts the probabilistic Sharpe for the number of alternative
    strategies TESTED. Multiple testing inflates the observed max
    Sharpe; DSR back it out.

    Parameters
    ----------
    sharpe : float
        Observed Sharpe (usually the max across `n_trials`).
    n : int
        Number of return observations for the observed strategy.
    n_trials : int
        How many alternative strategies were tested.
    variance_of_sharpes : float
        Variance of the Sharpes across the trials.
    skew, kurt : float
        Return-series skewness and (raw) kurtosis.

    Returns
    -------
    float
        Probability that the observed Sharpe exceeds a null-appropriate
        benchmark, adjusted for the number of trials.

    References
    ----------
    Bailey & Lopez de Prado 2014, "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting, and Non-Normality".
    """
    try:
        from scipy.stats import norm
    except ImportError as e:
        require_dep("scipy", kernel="deflated_sharpe", install="pip install scipy", cause=e)
    require_positive(n, "n", kernel="deflated_sharpe", kind="int")
    require_positive(n_trials, "n_trials", kernel="deflated_sharpe", kind="int")
    # Expected max Sharpe under the null of zero true Sharpe:
    #   E[max] ≈ sqrt(V) * ((1 - γ) * Z^{-1}(1 - 1/N) + γ * Z^{-1}(1 - 1/(N e)))
    # with γ = Euler-Mascheroni.
    gamma_euler = 0.5772156649015329
    if n_trials < 2:
        expected_max = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / float(n_trials))
        z2 = norm.ppf(1.0 - 1.0 / (float(n_trials) * np.e))
        expected_max = float(
            np.sqrt(max(variance_of_sharpes, 0.0)) * ((1.0 - gamma_euler) * z1 + gamma_euler * z2)
        )
    return probabilistic_sharpe(
        sharpe=sharpe,
        n=n,
        skew=skew,
        kurt=kurt,
        sharpe_benchmark=expected_max,
    )


# ---------- top-N drawdown table --------------------------------------


@dataclass
class DrawdownTableResult:
    """Top-N drawdown episodes with peak / trough / recovery details.

    Attributes
    ----------
    peaks : list[int]
    troughs : list[int]
    recoveries : list[int or None]
        Position where the peak was reclaimed. None if still underwater.
    depths : list[float]
        Drawdown magnitude (negative number, e.g. -0.20 = 20% drawdown).
    durations : list[int]
        `trough - peak` in bars.
    recovery_times : list[int or None]
        `recovery - trough` in bars. None if not recovered.
    n : int
        Number of episodes returned (up to `top_n`).
    """

    peaks: list
    troughs: list
    recoveries: list
    depths: list
    durations: list
    recovery_times: list
    n: int

    def summary(self) -> str:
        lines = [
            "=== DrawdownTableResult ===",
            f"{'depth':>10s} {'peak':>6s} {'trough':>7s} {'dur':>5s} {'rec':>5s}",
        ]
        for i in range(self.n):
            rec = str(self.recovery_times[i]) if self.recovery_times[i] is not None else "-"
            lines.append(
                f"{self.depths[i]:>+10.4%} {self.peaks[i]:>6d} {self.troughs[i]:>7d} "
                f"{self.durations[i]:>5d} {rec:>5s}"
            )
        return "\n".join(lines)


def drawdown_table(equity, top_n: int = 5) -> DrawdownTableResult:
    """Detect and rank distinct drawdown episodes on an equity curve.

    Parameters
    ----------
    equity : 1D array
        Strictly positive equity values.
    top_n : int, default 5
        Return at most this many episodes, ranked by depth.

    Returns
    -------
    DrawdownTableResult

    Notes
    -----
    An episode is bracketed as: the equity leaves a running-max high,
    dips, and either recovers (reaches or exceeds the prior high) or
    stays underwater to the end of the series. Overlapping episodes
    are merged; only distinct trough-to-peak arcs are reported.
    """
    require_positive(top_n, "top_n", kernel="drawdown_table", kind="int")
    arr = np.asarray(equity, dtype=np.float64)
    require_1d(arr, "equity", kernel="drawdown_table")
    finite = np.isfinite(arr)
    if not bool(finite.any()) or bool((arr[finite] <= 0).any()):
        raise KuantValueError(
            "kuant.drawdown_table: 'equity' must be strictly positive.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: use np.cumprod(1 + returns) to convert returns to equity"
        )

    running_max = np.maximum.accumulate(np.where(finite, arr, -np.inf))
    dd_series = arr / np.where(running_max == -np.inf, np.nan, running_max) - 1.0

    # Walk the series to find episodes.
    episodes = []
    n = arr.size
    i = 0
    while i < n:
        # Skip until we find a bar below its running max.
        while i < n and (not np.isfinite(dd_series[i]) or dd_series[i] >= 0):
            i += 1
        if i >= n:
            break
        peak_pos = int(np.argmax(arr[:i] == running_max[i]))
        # Consume until we recover to running_max[peak_pos] or hit the end.
        peak_val = float(running_max[i])
        trough_pos = i
        trough_val = float(arr[i])
        j = i
        while j < n:
            if not np.isfinite(arr[j]):
                j += 1
                continue
            if arr[j] < trough_val:
                trough_val = float(arr[j])
                trough_pos = j
            if arr[j] >= peak_val:
                break
            j += 1
        recovery = j if (j < n and arr[j] >= peak_val) else None
        depth = trough_val / peak_val - 1.0
        episodes.append(
            {
                "peak": peak_pos,
                "trough": trough_pos,
                "recovery": recovery,
                "depth": float(depth),
                "duration": trough_pos - peak_pos,
                "recovery_time": (recovery - trough_pos) if recovery is not None else None,
            }
        )
        # Advance past this episode; move to the recovery bar (or end).
        i = j + 1 if recovery is not None else n

    # Sort by depth (most negative first) and keep top_n.
    episodes.sort(key=lambda e: e["depth"])
    kept = episodes[: int(top_n)]

    return DrawdownTableResult(
        peaks=[e["peak"] for e in kept],
        troughs=[e["trough"] for e in kept],
        recoveries=[e["recovery"] for e in kept],
        depths=[e["depth"] for e in kept],
        durations=[e["duration"] for e in kept],
        recovery_times=[e["recovery_time"] for e in kept],
        n=len(kept),
    )


__all__ = [
    "CaptureResult",
    "DrawdownTableResult",
    "UlcerResult",
    "deflated_sharpe",
    "down_capture",
    "drawdown_table",
    "kelly",
    "omega",
    "probabilistic_sharpe",
    "ulcer_index",
    "up_capture",
]

"""Peak-to-trough drawdown series and maximum drawdown.

For a running equity curve (typically `np.cumprod(1 + returns)` or
`np.exp(np.cumsum(log_returns))`), the drawdown at each bar is:

    drawdown[t] = equity[t] / max(equity[0..t]) - 1

Always ≤ 0. Zero at every new peak; negative in between. `max_dd`
is the most-negative value across the series.

For rolling max drawdown over a trailing window, use
`kuant.stats.rollmdd` instead. This kernel is the full-history
version.

Design: docs/kernels/portfolio/drawdown.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_dep, warn_kuant
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class DrawdownResult:
    """Per-bar drawdown series plus scalar summary metrics.

    Attributes
    ----------
    series : 1D np.ndarray, length T
        Drawdown at each bar. Always in `[-1, 0]`; zero at every new
        equity peak.
    max_dd : float
        The most negative value in `series`. Reported as a NEGATIVE
        number (e.g. -0.15 for a 15% drawdown).
    peak_position : int
        Row index of the peak that preceded the max drawdown.
    trough_position : int
        Row index of the trough.
    duration : int
        `trough_position - peak_position` in bars.
    recovered : bool
        True iff the equity curve later reached a new high after the
        trough. False if the curve is still under water at the end
        of the series.
    """

    series: np.ndarray
    max_dd: float
    peak_position: int
    trough_position: int
    duration: int
    recovered: bool

    def summary(self) -> str:
        parts = [
            "=== DrawdownResult ===",
            f"max drawdown:      {self.max_dd:+.4%}",
            f"peak position:     {self.peak_position}",
            f"trough position:   {self.trough_position}",
            f"duration:          {self.duration} bars",
            f"recovered:         {self.recovered}",
        ]
        return "\n".join(parts)

    def to_parquet(self, path) -> None:
        """Write the drawdown series to parquet. Requires pyarrow."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="drawdown.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        table = pa.table({"drawdown": pa.array(self.series)})
        pq.write_table(table, path)


def drawdown(equity) -> DrawdownResult:
    """Peak-to-trough drawdown series and maximum drawdown.

    Parameters
    ----------
    equity : 1D array
        Running equity curve. Values must be strictly positive; use
        `np.cumprod(1 + returns)` if you have periodic returns.

    Returns
    -------
    DrawdownResult
        `.series` (T,) drawdown at each bar, `.max_dd` scalar,
        `.peak_position`, `.trough_position`, `.duration`, `.recovered`.

    Notes
    -----
    - Rejects non-positive equity values. A zero or negative equity
      breaks the peak/max ratio semantics.
    - NaN in the input propagates: the drawdown at NaN bars is NaN,
      and NaN does not update the running max.

    Examples
    --------
    >>> import numpy as np
    >>> # Equity peaks at 110, troughs at 88, recovers to 105.
    >>> eq = np.array([100.0, 105, 110, 100, 88, 95, 105])
    >>> r = drawdown(eq)
    >>> round(r.max_dd, 4)
    -0.2
    >>> r.peak_position, r.trough_position
    (2, 4)
    >>> r.recovered
    False
    """
    arr = np.asarray(equity, dtype=np.float64)
    require_1d(arr, "equity", kernel="drawdown")
    if arr.size == 0:
        raise KuantValueError(
            "kuant.drawdown: 'equity' is empty; drawdown is undefined "
            "on a zero-length series.  [KE-VAL-EMPTY]\n"
            "  → Fix: pass at least one equity observation; typical "
            "inputs are `np.cumprod(1 + returns)` from a non-empty "
            "return series"
        )

    finite_mask = np.isfinite(arr)
    if bool(finite_mask.any()) and bool((arr[finite_mask] <= 0).any()):
        bad_idx = int(np.where(finite_mask & (arr <= 0))[0][0])
        raise KuantValueError(
            f"kuant.drawdown: 'equity' contains non-positive values; "
            f"index {bad_idx} = {float(arr[bad_idx])}.  [KE-VAL-POSITIVE]\n"
            f"  → Fix: the equity curve must stay strictly positive. "
            f"If you have returns, convert with "
            f"`np.cumprod(1 + returns)` (simple) or "
            f"`np.exp(np.cumsum(log_returns))` (log)"
        )

    n = arr.size

    # Running max that ignores NaN via a fill-forward. NaN bars neither
    # update the running max nor contribute a drawdown value.
    running_max = np.maximum.accumulate(np.where(finite_mask, arr, -np.inf))
    # Bars before the first finite value have running_max = -inf; replace
    # so the ratio is NaN there.
    running_max = np.where(running_max == -np.inf, np.nan, running_max)

    series = arr / running_max - 1.0

    # Locate the trough. For all-NaN series, everything is NaN.
    if not bool(np.isfinite(series).any()):
        warn_kuant(
            kernel="drawdown",
            code="KW-DRAWDOWN-ALL-NAN",
            what=(
                "no finite equity observations; drawdown series returned "
                "as all-NaN and summary fields are NaN"
            ),
            fix=(
                "check upstream: the equity curve arrived as NaN, usually "
                "because returns had NaN-fills that were fed to cumprod "
                "without a fillna"
            ),
            category=KuantNumericWarning,
        )
        return DrawdownResult(
            series=series,
            max_dd=float("nan"),
            peak_position=0,
            trough_position=0,
            duration=0,
            recovered=False,
        )

    trough_pos = int(np.nanargmin(series))
    max_dd = float(series[trough_pos])
    # Peak that fed this trough: the running-max value at trough_pos
    # is the equity peak; find the position of that peak.
    peak_val = running_max[trough_pos]
    # Argmax returns the FIRST index where equity reached that peak.
    peak_pos = int(np.argmax(arr == peak_val))
    duration = trough_pos - peak_pos
    # Recovered iff any equity value AFTER the trough reaches peak_val again.
    recovered = bool((arr[trough_pos + 1 :] >= peak_val).any()) if trough_pos + 1 < n else False

    return DrawdownResult(
        series=series,
        max_dd=max_dd,
        peak_position=peak_pos,
        trough_position=trough_pos,
        duration=duration,
        recovered=recovered,
    )


__all__ = ["drawdown", "DrawdownResult"]

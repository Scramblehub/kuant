"""Heuristic delisting detection when a real lifecycle table isn't available.

Real-world callers should feed `SecurityLifecycle` objects sourced from
an exchange's daily delistings feed or a vendor delisting-code table.
When none is at hand — yfinance-only, alpaca-only, or a scraped panel —
`detect_delistings` reads the panel itself and flags columns that have
gone permanently NaN.

The heuristic is deliberately simple:

    A symbol is treated as delisted at date `d` when
        d = the last non-NaN row of that column
    AND
        the column has at least `min_gap_days` NaN rows AFTER `d`
    AND
        the panel extends beyond `d + min_gap_days`.

Setting `min_gap_days=5` avoids flagging weekend / holiday / halt gaps
as delistings.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from kuant._validation import require_dep, require_positive
from kuant.errors import KuantShapeError
from kuant.backtest.lifecycle.security import (
    SecurityLifecycle,
    TerminalAction,
    _as_date,
)


def detect_delistings(prices, min_gap_days: int = 5) -> dict[str, date]:
    """Flag columns that go permanently NaN before the panel's end.

    Parameters
    ----------
    prices : pandas.DataFrame
        Columns are symbols; index is date-like.
    min_gap_days : int, default 5
        Minimum number of NaN rows AFTER the last real print required
        before we call a column delisted. Prevents weekend / holiday /
        multi-day-halt gaps from being mistaken for delistings.

    Returns
    -------
    dict[str, datetime.date]
        Mapping symbol → last valid date. Columns that are still
        printing at the end of the panel are absent from the mapping.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.date_range("2020-01-01", periods=20, freq="D")
    >>> df = pd.DataFrame({
    ...     "STILL_LIVE": np.arange(20, dtype=float),
    ...     "DELISTED":   list(range(10)) + [np.nan] * 10,
    ... }, index=idx)
    >>> out = detect_delistings(df, min_gap_days=5)
    >>> sorted(out.keys())
    ['DELISTED']
    """
    try:
        import pandas as pd
    except ImportError as e:
        require_dep(
            "pandas",
            kernel="detect_delistings",
            install="pip install pandas",
            cause=e,
        )
    if not isinstance(prices, pd.DataFrame):
        raise KuantShapeError(
            f"kuant.detect_delistings: 'prices' must be a pandas.DataFrame, "
            f"got {type(prices).__name__}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: wrap the panel with pd.DataFrame(...)"
        )
    require_positive(min_gap_days, "min_gap_days", kernel="detect_delistings", kind="int")

    n = len(prices)
    out: dict[str, date] = {}
    for col in prices.columns:
        series = prices[col]
        finite = series.notna()
        if not bool(finite.any()):
            continue
        last_idx = int(np.flatnonzero(finite.to_numpy())[-1])
        trailing_nan = n - 1 - last_idx
        if trailing_nan >= int(min_gap_days):
            out[str(col)] = _as_date(prices.index[last_idx])
    return out


def lifecycles_from_panel(
    prices,
    min_gap_days: int = 5,
    terminal_action: TerminalAction = TerminalAction.MARK_TO_ZERO,
) -> dict[str, SecurityLifecycle]:
    """Turn a heuristic detection into a ready-to-use lifecycle mapping.

    Convenience one-liner: detect delistings, wrap each into a
    `SecurityLifecycle` with the same terminal action, ready to feed
    into `apply_lifecycle_panel` or `lifecycle_panel_report`.

    Parameters
    ----------
    prices : pandas.DataFrame
    min_gap_days : int, default 5
    terminal_action : TerminalAction, default MARK_TO_ZERO
        Assumed for every detected delisting. Users with per-symbol
        outcome data should construct their own dict directly.

    Returns
    -------
    dict[str, SecurityLifecycle]
    """
    detected = detect_delistings(prices, min_gap_days=min_gap_days)
    return {
        sym: SecurityLifecycle(
            symbol=sym,
            delisting_date=d,
            terminal_action=terminal_action,
        )
        for sym, d in detected.items()
    }


__all__ = ["detect_delistings", "lifecycles_from_panel"]

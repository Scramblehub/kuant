"""SecurityLifecycle: first-class listing / delisting semantics.

The gap this closes: most backtest engines (including vectorbt through
its current open-source line) silently ignore orders on NaN prices and
either drop the position or forward-fill the last live price forever.
On a real point-in-time equity book, both behaviors quietly corrupt
returns — the sign of long-short Sharpe can flip on any survivorship-
adjusted panel.

`SecurityLifecycle` is the ledger entry that says "this ticker was
listed here, delisted there, and here is what happened to a residual
position on the terminal date." Three terminal actions match the
distinctions vendors typically encode in a delisting-return code:

- `LIQUIDATE_AT_LAST` — sold at close on the delisting date. Optimistic;
  assumes fills exist at the reported last price.
- `MARK_TO_ZERO` — bankruptcy / worthless-close. Terminal-day-plus-one
  return is -1.0.
- `PRORATE_RECOVERY` — reorganization, forced conversion, cash-plus-
  stock merger with a known recovery ratio r ∈ [0, 1]. Terminal-day-
  plus-one return is r - 1.

Kernels:

- `apply_lifecycle(prices, lifecycle)` — mask pre-listing to NaN,
  cut prices to NaN after delisting.
- `apply_lifecycle_panel(prices_df, lifecycles)` — per-column dispatch.
- `lifecycle_returns(prices, lifecycle)` — returns with the terminal
  transition baked in.
- `tradeable_mask(index, lifecycle)` — boolean gate for simulators:
  True iff an order could have executed on that date.

Design: docs/kernels/lifecycle/security.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

import numpy as np

from kuant._validation import require_dep, require_probability
from kuant.errors import KuantShapeError, KuantValueError


class TerminalAction(str, Enum):
    """What happens to a position held on the delisting date.

    Values are strings so a lifecycle can round-trip through JSON /
    parquet without a custom encoder.
    """

    LIQUIDATE_AT_LAST = "liquidate_at_last"
    MARK_TO_ZERO = "mark_to_zero"
    PRORATE_RECOVERY = "prorate_recovery"


@dataclass(frozen=True)
class SecurityLifecycle:
    """Listing / delisting metadata for a single security.

    Attributes
    ----------
    symbol : str
        Ticker or identifier the caller uses in its price panel.
    listing_date : date or None
        First tradeable date. `None` means "listed before the price
        series begins" (no pre-listing masking).
    delisting_date : date or None
        Last tradeable date. `None` means "still trading."
    terminal_action : TerminalAction
        How to handle a position held on `delisting_date`. Ignored if
        `delisting_date` is None.
    terminal_recovery : float
        Recovery fraction in [0, 1]. Only consulted when
        `terminal_action == PRORATE_RECOVERY`.

    Examples
    --------
    >>> from datetime import date
    >>> lc = SecurityLifecycle(
    ...     symbol="ENRN",
    ...     delisting_date=date(2001, 12, 3),
    ...     terminal_action=TerminalAction.MARK_TO_ZERO,
    ... )
    >>> lc.symbol
    'ENRN'
    """

    symbol: str
    listing_date: date | None = None
    delisting_date: date | None = None
    terminal_action: TerminalAction = TerminalAction.MARK_TO_ZERO
    terminal_recovery: float = 0.0

    def __post_init__(self) -> None:
        require_probability(self.terminal_recovery, "terminal_recovery", kernel="SecurityLifecycle")
        if (
            self.listing_date is not None
            and self.delisting_date is not None
            and self.listing_date > self.delisting_date
        ):
            raise KuantValueError(
                f"kuant.SecurityLifecycle: 'listing_date' "
                f"({self.listing_date}) must be <= 'delisting_date' "
                f"({self.delisting_date}).  [KE-VAL-RANGE]\n"
                f"  → Fix: swap them or set the wrong one to None"
            )

    def summary(self) -> str:
        return (
            "=== SecurityLifecycle ===\n"
            f"symbol:             {self.symbol}\n"
            f"listing_date:       {self.listing_date}\n"
            f"delisting_date:     {self.delisting_date}\n"
            f"terminal_action:    {self.terminal_action.value}\n"
            f"terminal_recovery:  {self.terminal_recovery:.4f}"
        )


# ---------- index helpers ----------------------------------------------


def _as_date(x) -> date:
    """Coerce a variety of date-like objects to `datetime.date`."""
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    # numpy.datetime64 / pandas.Timestamp
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None  # type: ignore[assignment]
    if pd is not None:
        try:
            ts = pd.Timestamp(x)
            return ts.date()
        except Exception:
            pass
    # Fallback: numpy.datetime64
    if isinstance(x, np.datetime64):
        return np.datetime64(x, "D").astype("O")
    raise KuantValueError(
        f"kuant.lifecycle: could not coerce {x!r} of type "
        f"{type(x).__name__} to a date.  [KE-VAL-TYPE]\n"
        f"  → Fix: pass a datetime.date, pandas.Timestamp, or numpy.datetime64"
    )


def _index_to_dates(index) -> list[date]:
    """Convert a pandas / numpy index-like to a list of `datetime.date`."""
    try:
        return [_as_date(x) for x in index]
    except Exception as exc:
        raise KuantValueError(
            "kuant.lifecycle: could not coerce index to dates.  "
            "[KE-VAL-TYPE]\n"
            "  → Fix: pass a DatetimeIndex or a sequence of dates"
        ) from exc


# ---------- tradeable_mask ---------------------------------------------


def tradeable_mask(index, lifecycle: SecurityLifecycle) -> np.ndarray:
    """Boolean mask: True where an order could have executed.

    Parameters
    ----------
    index : sequence of dates (pandas DatetimeIndex, list of date, ...)
    lifecycle : SecurityLifecycle

    Returns
    -------
    1D np.ndarray[bool] of the same length as `index`.

    Notes
    -----
    Semantics:
      - `t < listing_date` → False
      - `listing_date <= t <= delisting_date` → True
      - `t > delisting_date` → False

    Simulators should use this to gate order fills, NOT rely on NaN
    prices — vectorbt's silent-NaN behavior is what this exists to
    replace.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import date
    >>> idx = pd.date_range("2020-01-01", periods=5, freq="D")
    >>> lc = SecurityLifecycle(
    ...     symbol="X",
    ...     listing_date=date(2020, 1, 2),
    ...     delisting_date=date(2020, 1, 4),
    ... )
    >>> tradeable_mask(idx, lc).tolist()
    [False, True, True, True, False]
    """
    dates = _index_to_dates(index)
    n = len(dates)
    mask = np.ones(n, dtype=bool)
    if lifecycle.listing_date is not None:
        lo = lifecycle.listing_date
        mask &= np.array([d >= lo for d in dates])
    if lifecycle.delisting_date is not None:
        hi = lifecycle.delisting_date
        mask &= np.array([d <= hi for d in dates])
    return mask


# ---------- apply_lifecycle --------------------------------------------


def apply_lifecycle(prices, lifecycle: SecurityLifecycle):
    """Mask a price series to the lifecycle's tradeable window.

    Parameters
    ----------
    prices : pandas.Series
        Index must be date-like.
    lifecycle : SecurityLifecycle

    Returns
    -------
    pandas.Series
        Same index. Pre-listing and post-delisting rows set to NaN.
        Values within the tradeable window are the original prices
        untouched.

    Notes
    -----
    This does NOT insert a synthetic terminal return; that is
    `lifecycle_returns`'s job. Cleaned prices are the substrate other
    kuant kernels consume.
    """
    try:
        import pandas as pd
    except ImportError as e:
        require_dep(
            "pandas",
            kernel="apply_lifecycle",
            install="pip install pandas",
            cause=e,
        )
    if not isinstance(prices, pd.Series):
        raise KuantShapeError(
            f"kuant.apply_lifecycle: 'prices' must be a pandas.Series, "
            f"got {type(prices).__name__}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: for panels use `apply_lifecycle_panel`; for raw "
            f"arrays wrap in `pd.Series(arr, index=dates)` first"
        )
    mask = tradeable_mask(prices.index, lifecycle)
    out = prices.astype(np.float64).copy()
    out[~mask] = np.nan
    return out


def apply_lifecycle_panel(prices, lifecycles: Mapping[str, SecurityLifecycle]):
    """Apply lifecycles column-by-column across a price panel.

    Parameters
    ----------
    prices : pandas.DataFrame
        Columns are ticker symbols; index is date-like.
    lifecycles : mapping symbol → SecurityLifecycle
        Columns missing from the mapping are left untouched. Extra
        entries in the mapping (symbols not in `prices.columns`) are
        ignored.

    Returns
    -------
    pandas.DataFrame
        Same shape as `prices`, with lifecycle-masked columns.

    Notes
    -----
    Deliberately does NOT reindex or add columns. If a lifecycle names
    a symbol not present, we do not silently synthesize an empty
    column — that would mask upstream data-loading bugs.
    """
    try:
        import pandas as pd
    except ImportError as e:
        require_dep(
            "pandas",
            kernel="apply_lifecycle_panel",
            install="pip install pandas",
            cause=e,
        )
    if not isinstance(prices, pd.DataFrame):
        raise KuantShapeError(
            f"kuant.apply_lifecycle_panel: 'prices' must be a "
            f"pandas.DataFrame, got {type(prices).__name__}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: for a single symbol use `apply_lifecycle` on a "
            f"pandas.Series"
        )
    out = prices.astype(np.float64).copy()
    for sym, lc in lifecycles.items():
        if sym not in out.columns:
            continue
        mask = tradeable_mask(out.index, lc)
        out.loc[~mask, sym] = np.nan
    return out


# ---------- lifecycle_returns ------------------------------------------


def lifecycle_returns(prices, lifecycle: SecurityLifecycle):
    """Returns with the terminal transition baked in.

    Parameters
    ----------
    prices : pandas.Series
        Raw (un-masked) prices. Only rows within the lifecycle window
        are used for the ordinary-return calculation.
    lifecycle : SecurityLifecycle

    Returns
    -------
    pandas.Series
        Same index as `prices`.
        - Pre-listing rows: NaN.
        - First in-window row: NaN (no prior close for a return).
        - In-window rows: pct_change of in-window prices.
        - One row AFTER `delisting_date` if such a row exists in the
          index: the terminal return (0 for LIQUIDATE_AT_LAST, -1.0
          for MARK_TO_ZERO, `recovery - 1` for PRORATE_RECOVERY).
        - All later rows: NaN.

    Notes
    -----
    Convention: LIQUIDATE_AT_LAST leaves the terminal-day-plus-one
    return at 0.0, NOT NaN — the semantic being "position closed
    at close, cash held, no move." Callers who don't want that row
    at all can drop rows where `tradeable_mask` is False AFTER the
    terminal return.
    """
    try:
        import pandas as pd
    except ImportError as e:
        require_dep(
            "pandas",
            kernel="lifecycle_returns",
            install="pip install pandas",
            cause=e,
        )
    if not isinstance(prices, pd.Series):
        raise KuantShapeError(
            f"kuant.lifecycle_returns: 'prices' must be a pandas.Series, "
            f"got {type(prices).__name__}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: wrap raw arrays in `pd.Series(arr, index=dates)`"
        )
    mask = tradeable_mask(prices.index, lifecycle)
    p = prices.astype(np.float64)
    in_window = p.where(mask)
    ret = in_window.pct_change(fill_method=None)
    # First in-window row's pct_change against a pre-listing NaN comes
    # out NaN already; nothing extra to do there.
    # Terminal row: the first index position AFTER delisting_date.
    if lifecycle.delisting_date is not None:
        dates = _index_to_dates(prices.index)
        after = [i for i, d in enumerate(dates) if d > lifecycle.delisting_date]
        if after:
            j = after[0]
            action = lifecycle.terminal_action
            if action == TerminalAction.LIQUIDATE_AT_LAST:
                ret.iloc[j] = 0.0
            elif action == TerminalAction.MARK_TO_ZERO:
                ret.iloc[j] = -1.0
            elif action == TerminalAction.PRORATE_RECOVERY:
                ret.iloc[j] = float(lifecycle.terminal_recovery) - 1.0
            else:  # pragma: no cover
                raise KuantValueError(
                    f"kuant.lifecycle_returns: unknown terminal_action "
                    f"{action!r}.  [KE-VAL-RANGE]"
                )
            # Rows strictly after the terminal row: force NaN.
            if j + 1 < len(ret):
                ret.iloc[j + 1 :] = np.nan
    return ret


# ---------- LifecyclePanelResult ---------------------------------------


@dataclass
class LifecyclePanelResult:
    """Combined output of `lifecycle_panel_report` on a full panel.

    Attributes
    ----------
    cleaned : pandas.DataFrame
        Prices masked to each column's tradeable window.
    tradeable : pandas.DataFrame
        Boolean panel: True where an order could have filled.
    terminal_returns : pandas.DataFrame
        Same shape as `cleaned`. All zero except the single terminal
        row for each column, which carries the action-specific return.
        Useful for reconciling total-return calculations.
    lifecycles : dict[str, SecurityLifecycle]
        Copy of the mapping used to build this result.
    """

    cleaned: object  # pandas.DataFrame
    tradeable: object  # pandas.DataFrame
    terminal_returns: object  # pandas.DataFrame
    lifecycles: dict = field(default_factory=dict)

    def summary(self) -> str:
        n_delisted = sum(1 for lc in self.lifecycles.values() if lc.delisting_date is not None)
        return (
            "=== LifecyclePanelResult ===\n"
            f"panel shape:        {self.cleaned.shape}\n"
            f"n symbols:          {len(self.lifecycles)}\n"
            f"n with delisting:   {n_delisted}"
        )

    def to_parquet(self, path) -> None:
        """Write the cleaned panel to parquet. Requires pyarrow.

        Only the cleaned prices are serialized. Reconstruct the mask
        and terminal-return frames by re-running the kernels against
        the original lifecycle map.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="LifecyclePanelResult.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        cols = {"row_index": pa.array([str(x) for x in self.cleaned.index])}
        for name in self.cleaned.columns:
            cols[str(name)] = pa.array(self.cleaned[name].to_numpy())
        pq.write_table(pa.table(cols), path)


def lifecycle_panel_report(
    prices, lifecycles: Mapping[str, SecurityLifecycle]
) -> LifecyclePanelResult:
    """Bundle cleaned + tradeable + terminal-return frames.

    Convenience wrapper the user reaches for when they want everything
    the primitive can offer for a panel in one call.

    Parameters
    ----------
    prices : pandas.DataFrame
    lifecycles : mapping symbol → SecurityLifecycle

    Returns
    -------
    LifecyclePanelResult
    """
    try:
        import pandas as pd
    except ImportError as e:
        require_dep(
            "pandas",
            kernel="lifecycle_panel_report",
            install="pip install pandas",
            cause=e,
        )
    if not isinstance(prices, pd.DataFrame):
        raise KuantShapeError(
            f"kuant.lifecycle_panel_report: 'prices' must be a "
            f"pandas.DataFrame, got {type(prices).__name__}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: convert with pd.DataFrame(...)"
        )
    cleaned = apply_lifecycle_panel(prices, lifecycles)
    tradeable = pd.DataFrame(
        {sym: tradeable_mask(prices.index, lc) for sym, lc in lifecycles.items()},
        index=prices.index,
    )
    term = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for sym, lc in lifecycles.items():
        if sym not in prices.columns or lc.delisting_date is None:
            continue
        dates = _index_to_dates(prices.index)
        after = [i for i, d in enumerate(dates) if d > lc.delisting_date]
        if not after:
            continue
        j = after[0]
        if lc.terminal_action == TerminalAction.LIQUIDATE_AT_LAST:
            term.iloc[j, term.columns.get_loc(sym)] = 0.0
        elif lc.terminal_action == TerminalAction.MARK_TO_ZERO:
            term.iloc[j, term.columns.get_loc(sym)] = -1.0
        elif lc.terminal_action == TerminalAction.PRORATE_RECOVERY:
            term.iloc[j, term.columns.get_loc(sym)] = float(lc.terminal_recovery) - 1.0
    return LifecyclePanelResult(
        cleaned=cleaned,
        tradeable=tradeable,
        terminal_returns=term,
        lifecycles=dict(lifecycles),
    )


__all__ = [
    "TerminalAction",
    "SecurityLifecycle",
    "LifecyclePanelResult",
    "apply_lifecycle",
    "apply_lifecycle_panel",
    "lifecycle_returns",
    "lifecycle_panel_report",
    "tradeable_mask",
]

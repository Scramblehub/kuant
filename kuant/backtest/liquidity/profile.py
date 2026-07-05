"""LiquidityProfile: per-security volume / spread metadata for fills.

A profile bundles the market-microstructure inputs a fill model needs
to answer "can this order fill at the stated price, and if so, how
much slippage?" for a specific symbol on a specific date:

- `adv_series`: average daily volume in shares (or contracts), indexed
  by date.
- `spread_series`: bid-ask spread in basis points, indexed by date.
  Optional. Set to None if the fill model does not consume it.
- `min_size`: minimum lot size in the same units as `adv_series`. An
  order below this size is rejected outright, not silently truncated.
- `max_participation`: fraction of ADV a single order can consume in
  one bar. Orders larger than `max_participation * ADV` are truncated
  to the cap. Default 0.10 (10% ADV), a common institutional guideline.

Design: docs/kernels/backtest/liquidity/profile.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from kuant._validation import require_positive, require_range
from kuant.errors import KuantShapeError, KuantValueError


@dataclass(frozen=True)
class LiquidityProfile:
    """Volume + spread metadata for a single security.

    Attributes
    ----------
    symbol : str
        The identifier the caller uses in its price and volume panels.
    adv_series : pandas.Series
        Average daily volume per date. Index must be date-like.
    spread_series : pandas.Series or None
        Bid-ask spread in basis points per date. Same index as
        `adv_series` if provided.
    min_size : float
        Minimum order size (in the same units as `adv_series`). Orders
        below this are rejected, not silently rounded.
    max_participation : float
        Fraction in (0, 1] of ADV a single order can consume. Orders
        above `max_participation * ADV_t` are truncated to the cap
        with a `CAPPED_PARTICIPATION` reason on the fill result.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2020-01-01", periods=3, freq="D")
    >>> adv = pd.Series([1_000_000.0, 900_000.0, 1_200_000.0], index=idx)
    >>> profile = LiquidityProfile(
    ...     symbol="XYZ",
    ...     adv_series=adv,
    ...     spread_series=None,
    ...     min_size=100.0,
    ...     max_participation=0.05,
    ... )
    >>> profile.symbol
    'XYZ'
    """

    symbol: str
    adv_series: object  # pandas.Series
    spread_series: object = None  # pandas.Series or None
    min_size: float = 1.0
    max_participation: float = 0.10

    def __post_init__(self) -> None:
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            raise KuantValueError(
                "kuant.LiquidityProfile requires pandas.  "
                "[KE-DEP-MISSING]\n"
                "  → Fix: `pip install pandas`"
            ) from None
        if not isinstance(self.adv_series, pd.Series):
            raise KuantShapeError(
                f"kuant.LiquidityProfile: 'adv_series' must be a "
                f"pandas.Series, got {type(self.adv_series).__name__}.  "
                f"[KE-SHAPE-EXPECTED]\n"
                f"  → Fix: wrap the ADV data in `pd.Series(values, "
                f"index=dates)` before constructing the profile"
            )
        if self.spread_series is not None and not isinstance(self.spread_series, pd.Series):
            raise KuantShapeError(
                f"kuant.LiquidityProfile: 'spread_series' must be a "
                f"pandas.Series or None, got "
                f"{type(self.spread_series).__name__}.  "
                f"[KE-SHAPE-EXPECTED]\n"
                f"  → Fix: pass None if the fill model does not need "
                f"spread; otherwise wrap in `pd.Series(...)`"
            )
        if self.spread_series is not None:
            if len(self.spread_series) != len(self.adv_series):
                raise KuantShapeError(
                    f"kuant.LiquidityProfile: 'spread_series' length "
                    f"{len(self.spread_series)} does not match "
                    f"'adv_series' length {len(self.adv_series)}.  "
                    f"[KE-SHAPE-EQUAL-LEN]\n"
                    f"  → Fix: reindex both series onto a common date "
                    f"index before constructing the profile"
                )
        require_positive(self.min_size, "min_size", kernel="LiquidityProfile")
        require_range(
            self.max_participation,
            "max_participation",
            kernel="LiquidityProfile",
            lo=0.0,
            hi=1.0,
            lo_inclusive=False,
            hi_inclusive=True,
        )

    def summary(self) -> str:
        n = len(self.adv_series)
        has_spread = self.spread_series is not None
        return (
            "=== LiquidityProfile ===\n"
            f"symbol:              {self.symbol}\n"
            f"n rows:              {n}\n"
            f"min_size:            {self.min_size:g}\n"
            f"max_participation:   {self.max_participation:.4f}\n"
            f"has spread_series:   {has_spread}"
        )


__all__ = ["LiquidityProfile"]

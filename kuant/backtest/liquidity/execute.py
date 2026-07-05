"""Order execution against a LiquidityProfile.

The `execute_fill` kernel translates a requested (size, price) order
into an actual fill, respecting three real-world frictions vectorbt's
open-source line does not model:

1. **Participation cap.** An order larger than `max_participation *
   ADV_t` is truncated, not silently ignored. The rejected quantity is
   reported so a simulator can queue it into the next bar or drop it.
2. **Minimum lot size.** An order below `min_size` is refused outright.
   Fractional-share brokers relax this; institutional wires do not.
3. **Size-dependent slippage.** The fill model produces a slippage
   fraction from the participation rate, not a constant haircut.

The fill result carries the observed fill price, size filled, size
rejected, applied slippage in basis points, and a categorical `reason`
so downstream reporting can distinguish a partial fill from a full
rejection.

Design: docs/kernels/backtest/liquidity/execute.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kuant.backtest.lifecycle.security import _as_date
from kuant.backtest.liquidity.profile import LiquidityProfile
from kuant.errors import KuantShapeError, KuantValueError


# ---------- FillResult -------------------------------------------------


@dataclass
class FillResult:
    """Outcome of a single order attempt.

    Attributes
    ----------
    price : float
        Actual fill price after slippage. NaN if the order was rejected.
    size_filled : float
        Signed quantity filled (+buy, -sell). Zero if fully rejected.
    size_rejected : float
        Signed quantity NOT filled. Simulators may re-queue this or
        drop it depending on strategy.
    slippage_bps : float
        Applied slippage in basis points. Zero if fully rejected.
    reason : str
        Categorical outcome:
          - `"OK"`                    fully filled
          - `"CAPPED_PARTICIPATION"`  partially filled at the ADV cap
          - `"BELOW_MIN_SIZE"`        fully rejected: below min_size
          - `"NO_LIQUIDITY"`          fully rejected: ADV is NaN or zero
          - `"MISSING_DATE"`          fully rejected: timestamp not in
                                       profile's index
    cost : float
        Total transaction value = `abs(size_filled) * price`. Zero if
        rejected.
    """

    price: float
    size_filled: float
    size_rejected: float
    slippage_bps: float
    reason: str
    cost: float

    def summary(self) -> str:
        return (
            "=== FillResult ===\n"
            f"reason:         {self.reason}\n"
            f"price:          {self.price:.6f}\n"
            f"size filled:    {self.size_filled:+.4f}\n"
            f"size rejected:  {self.size_rejected:+.4f}\n"
            f"slippage bps:   {self.slippage_bps:+.4f}\n"
            f"cost:           {self.cost:.4f}"
        )


# ---------- execute_fill (single order) --------------------------------


def execute_fill(
    size: float,
    price: float,
    profile: LiquidityProfile,
    timestamp,
    model,
) -> FillResult:
    """Attempt to fill a single order against a liquidity profile.

    Parameters
    ----------
    size : float
        Signed order size. Positive = buy, negative = sell. In the same
        units as `profile.adv_series`.
    price : float
        Reference price (typically mid or close) before slippage.
    profile : LiquidityProfile
    timestamp : date, pd.Timestamp, or np.datetime64
        The date to look up in `profile.adv_series`.
    model : FillModel-like
        Any object exposing `compute_slippage(size, adv, side) -> float`.
        The bundled models are `FlatSlippage`, `LinearImpact`, and
        `SquareRootImpact`.

    Returns
    -------
    FillResult

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import date
    >>> from kuant.backtest.liquidity import (
    ...     LiquidityProfile, FlatSlippage, execute_fill,
    ... )
    >>> adv = pd.Series([1_000_000.0], index=[pd.Timestamp("2020-01-02")])
    >>> profile = LiquidityProfile(
    ...     symbol="XYZ", adv_series=adv, min_size=100.0,
    ...     max_participation=0.1,
    ... )
    >>> r = execute_fill(1000.0, 50.0, profile, date(2020, 1, 2),
    ...                  FlatSlippage(bps=5))
    >>> r.reason
    'OK'
    >>> round(r.price, 4)
    50.025
    """
    if not hasattr(model, "compute_slippage"):
        raise KuantValueError(
            f"kuant.execute_fill: 'model' must expose a "
            f"compute_slippage(size, adv, side) method, got "
            f"{type(model).__name__}.  [KE-VAL-CONTRACT]\n"
            f"  → Fix: pass one of FlatSlippage / LinearImpact / "
            f"SquareRootImpact, or a custom class with a matching method"
        )
    if not np.isfinite(price) or price <= 0:
        raise KuantValueError(
            f"kuant.execute_fill: 'price' must be positive and finite, "
            f"got {price}.  [KE-VAL-POSITIVE]\n"
            f"  → Fix: gate zero and NaN prices out with "
            f"kuant.backtest.lifecycle.tradeable_mask before calling"
        )

    ts = _as_date(timestamp)
    adv_series = profile.adv_series
    adv_dates = [_as_date(x) for x in adv_series.index]
    if ts not in adv_dates:
        return FillResult(
            price=float("nan"),
            size_filled=0.0,
            size_rejected=float(size),
            slippage_bps=0.0,
            reason="MISSING_DATE",
            cost=0.0,
        )
    adv = float(adv_series.iloc[adv_dates.index(ts)])
    if not np.isfinite(adv) or adv <= 0:
        return FillResult(
            price=float("nan"),
            size_filled=0.0,
            size_rejected=float(size),
            slippage_bps=0.0,
            reason="NO_LIQUIDITY",
            cost=0.0,
        )

    side = 1 if size > 0 else -1
    abs_req = abs(float(size))
    cap = profile.max_participation * adv

    if abs_req < profile.min_size:
        return FillResult(
            price=float("nan"),
            size_filled=0.0,
            size_rejected=float(size),
            slippage_bps=0.0,
            reason="BELOW_MIN_SIZE",
            cost=0.0,
        )

    if abs_req > cap:
        # Truncate to cap.
        abs_filled = cap
        reason = "CAPPED_PARTICIPATION"
    else:
        abs_filled = abs_req
        reason = "OK"

    slippage_frac = float(model.compute_slippage(abs_filled, adv, side))
    fill_price = price * (1.0 + side * slippage_frac)

    size_filled = side * abs_filled
    size_rejected = float(size) - size_filled
    return FillResult(
        price=float(fill_price),
        size_filled=float(size_filled),
        size_rejected=float(size_rejected),
        slippage_bps=float(slippage_frac * 10_000.0),
        reason=reason,
        cost=float(abs_filled * fill_price),
    )


# ---------- execute_fill_panel (vectorized batch) ---------------------


def execute_fill_panel(
    orders,
    profile: LiquidityProfile,
    model,
):
    """Vectorized fill over a batch of orders for one profile.

    Parameters
    ----------
    orders : pandas.DataFrame
        Columns required: `timestamp` (date-like), `size` (signed),
        `price` (positive). Row order determines output row order.
    profile : LiquidityProfile
    model : FillModel-like

    Returns
    -------
    pandas.DataFrame
        Columns: `timestamp`, `size` (requested), `price` (reference),
        `fill_price`, `size_filled`, `size_rejected`, `slippage_bps`,
        `reason`, `cost`.

    Notes
    -----
    Straightforward per-row loop over `execute_fill` in v1. A future
    version will vectorize the ADV lookup and slippage computation for
    speed; the API stays the same.
    """
    if not isinstance(orders, pd.DataFrame):
        raise KuantShapeError(
            f"kuant.execute_fill_panel: 'orders' must be a "
            f"pandas.DataFrame, got {type(orders).__name__}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: build a DataFrame with columns "
            f"['timestamp', 'size', 'price']"
        )
    required = {"timestamp", "size", "price"}
    missing = required - set(orders.columns)
    if missing:
        raise KuantValueError(
            f"kuant.execute_fill_panel: 'orders' is missing columns "
            f"{sorted(missing)}.  [KE-VAL-SCHEMA]\n"
            f"  → Fix: include all of {sorted(required)}"
        )

    rows = []
    for _, row in orders.iterrows():
        r = execute_fill(
            size=float(row["size"]),
            price=float(row["price"]),
            profile=profile,
            timestamp=row["timestamp"],
            model=model,
        )
        rows.append(
            {
                "timestamp": row["timestamp"],
                "size": float(row["size"]),
                "price": float(row["price"]),
                "fill_price": r.price,
                "size_filled": r.size_filled,
                "size_rejected": r.size_rejected,
                "slippage_bps": r.slippage_bps,
                "reason": r.reason,
                "cost": r.cost,
            }
        )
    return pd.DataFrame(rows)


# ---------- liquidity_mask --------------------------------------------


def liquidity_mask(index, profile: LiquidityProfile, min_adv: float = 0.0):
    """Boolean mask: True where ADV meets the minimum threshold.

    Parameters
    ----------
    index : sequence of dates
    profile : LiquidityProfile
    min_adv : float, default 0.0
        Minimum ADV per date required to trade. Defaults to 0, which
        only masks out NaN / non-positive ADV. Set to a higher value
        to filter thin-liquidity days.

    Returns
    -------
    1D np.ndarray[bool] of the same length as `index`.

    Notes
    -----
    Compose with `kuant.backtest.lifecycle.tradeable_mask` to get a
    full "can this order fill" gate:

        from kuant.backtest.lifecycle import tradeable_mask
        from kuant.backtest.liquidity import liquidity_mask
        can_trade = tradeable_mask(idx, lc) & liquidity_mask(idx, profile)
    """
    profile_dates = [_as_date(x) for x in profile.adv_series.index]
    profile_lookup = dict(zip(profile_dates, profile.adv_series.values))
    dates = [_as_date(x) for x in index]
    mask = np.zeros(len(dates), dtype=bool)
    for i, d in enumerate(dates):
        adv = profile_lookup.get(d)
        if adv is None:
            continue
        if not np.isfinite(adv):
            continue
        if adv > float(min_adv):
            mask[i] = True
    return mask


__all__ = [
    "FillResult",
    "execute_fill",
    "execute_fill_panel",
    "liquidity_mask",
]

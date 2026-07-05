"""submit_order: route an Order through a liquidity profile to a FillReport.

Thin adapter between the Order abstraction and `liquidity.execute_fill`.
The engine (arriving in a later session) will hold a queue of pending
orders; submit_order is what it calls per bar to actually execute one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kuant.backtest.fill.order import Order, OrderStatus, OrderType
from kuant.backtest.liquidity import FillResult, LiquidityProfile, execute_fill
from kuant.errors import KuantValueError


@dataclass
class FillReport:
    """Post-submission record for a single order.

    Wraps the underlying `FillResult` with tracking metadata so a
    portfolio can reconcile fills back to their originating orders.

    Attributes
    ----------
    order_id : int
        Copied from the submitted Order.
    symbol : str
    status : OrderStatus
        FILLED, PARTIALLY_FILLED, or REJECTED. PENDING is only used by
        an order queue before submission.
    fill : FillResult
        The underlying fill from `liquidity.execute_fill`.
    tag : str
        Carried through from the Order for downstream reporting.
    """

    order_id: int
    symbol: str
    status: OrderStatus
    fill: FillResult
    tag: str = field(default="")

    def summary(self) -> str:
        return (
            "=== FillReport ===\n"
            f"order_id:      {self.order_id}\n"
            f"symbol:        {self.symbol}\n"
            f"status:        {self.status.value}\n"
            f"tag:           {self.tag!r}\n"
            f"{self.fill.summary()}"
        )


def _status_from_reason(reason: str) -> OrderStatus:
    if reason == "OK":
        return OrderStatus.FILLED
    if reason == "CAPPED_PARTICIPATION":
        return OrderStatus.PARTIALLY_FILLED
    # BELOW_MIN_SIZE, NO_LIQUIDITY, MISSING_DATE → REJECTED.
    return OrderStatus.REJECTED


def submit_order(
    order: Order,
    profile: LiquidityProfile,
    price: float,
    model,
) -> FillReport:
    """Submit a MARKET order against a liquidity profile.

    Parameters
    ----------
    order : Order
        Must have `order_type == OrderType.MARKET` in v1.
    profile : LiquidityProfile
        The security's liquidity metadata. Must match `order.symbol`.
    price : float
        Reference price at submission (typically bar close).
    model : FillModel-like
        Any object exposing `compute_slippage(size, adv, side) -> float`.

    Returns
    -------
    FillReport

    Notes
    -----
    Non-MARKET orders raise `KuantValueError` in v1. Engine-level
    scheduling of LIMIT / STOP orders is deferred to the engine
    session.

    Symbol mismatch between `order.symbol` and `profile.symbol` is
    a hard error because it usually indicates a routing bug in the
    caller.
    """
    if order.order_type is not OrderType.MARKET:
        raise KuantValueError(
            f"kuant.submit_order: order_type={order.order_type.value} "
            f"is not executable in v1 (MARKET only).  "
            f"[KE-VAL-CONTRACT]\n"
            f"  → Fix: use OrderType.MARKET, or hold the order in a "
            f"queue until the engine's LIMIT/STOP layer ships"
        )
    if order.symbol != profile.symbol:
        raise KuantValueError(
            f"kuant.submit_order: order.symbol={order.symbol!r} does "
            f"not match profile.symbol={profile.symbol!r}.  "
            f"[KE-VAL-CONTRACT]\n"
            f"  → Fix: route each order to its matching profile"
        )

    fill = execute_fill(
        size=order.signed_size,
        price=price,
        profile=profile,
        timestamp=order.timestamp,
        model=model,
    )
    return FillReport(
        order_id=order.order_id,
        symbol=order.symbol,
        status=_status_from_reason(fill.reason),
        fill=fill,
        tag=order.tag,
    )


__all__ = ["FillReport", "submit_order"]

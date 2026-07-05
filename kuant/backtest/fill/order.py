"""Order primitives: Order dataclass + Side / Type / Status enums.

Explicit side and positive size, not signed-size. Keeps the direction
of a trade a first-class field so accidental sign flips in strategy
code are impossible.

v1 order types: MARKET only. `LIMIT` and `STOP` are reserved on the
enum for engine-level extension; `submit_order` currently rejects
anything other than MARKET.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from itertools import count as _count

from kuant._validation import require_positive
from kuant.errors import KuantValueError


class OrderSide(int, Enum):
    """Direction of a trade. Values are +1 / -1 for arithmetic use."""

    BUY = 1
    SELL = -1


class OrderType(str, Enum):
    """Fill semantics. Only MARKET is executable in v1."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    """Terminal or in-flight state of an order after submission."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELED = "canceled"


_order_id_seq = _count(1)


def _next_order_id() -> int:
    return next(_order_id_seq)


@dataclass
class Order:
    """A single order intent, not yet submitted.

    Attributes
    ----------
    symbol : str
    side : OrderSide
        BUY (+1) or SELL (-1).
    size : float
        Unsigned quantity in ADV units (typically shares). Must be > 0.
    order_type : OrderType, default MARKET
    limit_price : float or None
        Required for LIMIT / STOP. Ignored for MARKET.
    timestamp : datetime.date
        The date the order is submitted (typically a bar's close date).
    order_id : int
        Auto-assigned monotonically increasing. Callers can override
        for deterministic replay.
    tag : str
        Free-form label (strategy name, cohort, etc.). Not consumed by
        kernels; carried on the FillReport for reporting.
    """

    symbol: str
    side: OrderSide
    size: float
    timestamp: date
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    order_id: int = field(default_factory=_next_order_id)
    tag: str = ""

    def __post_init__(self) -> None:
        require_positive(self.size, "size", kernel="Order")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP):
            if self.limit_price is None:
                raise KuantValueError(
                    f"kuant.Order: order_type={self.order_type.value} "
                    f"requires a limit_price.  [KE-VAL-CONTRACT]\n"
                    f"  → Fix: pass limit_price=<float>"
                )
            import math

            if not math.isfinite(self.limit_price) or self.limit_price <= 0.0:
                raise KuantValueError(
                    f"kuant.Order: limit_price must be finite and "
                    f"strictly positive for {self.order_type.value} "
                    f"orders, got {self.limit_price}.  "
                    f"[KE-ORDER-LIMIT-INVALID]\n"
                    f"  → Fix: compute a real limit level from a "
                    f"reference price; None-or-positive is the only "
                    f"valid combination"
                )

    @property
    def signed_size(self) -> float:
        """Size with sign implied by side. Positive = buy, negative = sell."""
        return float(self.side.value) * float(self.size)

    def summary(self) -> str:
        return (
            "=== Order ===\n"
            f"order_id:      {self.order_id}\n"
            f"symbol:        {self.symbol}\n"
            f"side:          {self.side.value:+d}\n"
            f"size:          {self.size:g}\n"
            f"order_type:    {self.order_type.value}\n"
            f"limit_price:   {self.limit_price}\n"
            f"timestamp:     {self.timestamp}\n"
            f"tag:           {self.tag!r}"
        )


__all__ = ["Order", "OrderSide", "OrderStatus", "OrderType"]

"""kuant.backtest.fill: Order abstraction + submission routing.

Sits between strategy code (which produces Orders) and the liquidity
layer (which produces FillResults). Keeps sign conventions explicit
via `OrderSide`, tracks per-order status via `OrderStatus`, and reports
back to the caller through `FillReport` for reconciliation.

Primitives:

- `OrderSide`, `OrderType`, `OrderStatus`: enums.
- `Order`: dataclass with symbol, side, unsigned size, order_type,
  optional limit_price, timestamp, auto-assigned order_id, and a
  free-form tag.
- `FillReport`: wraps a `FillResult` with `order_id`, `symbol`,
  `status`, and `tag` for order reconciliation.
- `submit_order(order, profile, price, model)`: MARKET-only in v1.
  Routes the order through `kuant.backtest.liquidity.execute_fill`
  and translates the fill reason into a status.

Design: docs/kernels/backtest/fill/README.md.
"""

from kuant.backtest.fill.order import Order, OrderSide, OrderStatus, OrderType
from kuant.backtest.fill.submit import FillReport, submit_order

__all__ = [
    "FillReport",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "submit_order",
]

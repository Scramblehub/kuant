# kuant.backtest.fill

Order abstraction and submission routing for the backtest engine.

## Where this sits

`fill` is the thin layer between strategy code (which emits
`Order`s) and the liquidity layer (which produces `FillResult`s). A
strategy names a symbol, a side, an unsigned quantity, and a
timestamp; `submit_order` routes that intent through
`kuant.backtest.liquidity.execute_fill` and returns a `FillReport`
that a portfolio can reconcile back to the originating order.

Common backtest tooling folds these two responsibilities together:
the strategy hands the simulator a signed number, and slippage is a
constant on the engine object. That collapses two decisions
(direction of trade, quantity of trade) into one signed scalar with
easy-to-flip sign conventions, and it leaves no place to carry
per-order metadata like `order_id` or a strategy `tag` for later
attribution. `fill` splits them apart.

## Files

- [`order.md`](order.md): the `Order` dataclass and the
  `OrderSide` / `OrderType` / `OrderStatus` enums.
- [`submit.md`](submit.md): `submit_order` and the `FillReport`
  wrapper.

## Public API

```python
from kuant.backtest.fill import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    FillReport,
    submit_order,
)
```

Six things live here:

1. `Order`: dataclass with `symbol`, `side`, unsigned `size`,
   `timestamp`, `order_type`, optional `limit_price`,
   auto-assigned `order_id`, and free-form `tag`.
2. `OrderSide`: `BUY = +1`, `SELL = -1`. Explicit direction.
3. `OrderType`: `MARKET`, `LIMIT`, `STOP`. Only `MARKET` is
   executable in v1; the other two are reserved for the engine layer.
4. `OrderStatus`: `PENDING`, `FILLED`, `PARTIALLY_FILLED`,
   `REJECTED`, `CANCELED`.
5. `FillReport`: wraps a `FillResult` with `order_id`, `symbol`,
   `status`, and `tag` for reconciliation.
6. `submit_order(order, profile, price, model) -> FillReport`.

## Typical caller flow

```python
from datetime import date
from kuant.backtest.fill import Order, OrderSide, submit_order
from kuant.backtest.liquidity import LiquidityProfile, FlatSlippage

order = Order(
    symbol="XYZ",
    side=OrderSide.BUY,
    size=1000.0,
    timestamp=date(2020, 1, 2),
    tag="momentum-alpha",
)
report = submit_order(
    order,
    profile=profile,
    price=50.0,
    model=FlatSlippage(bps=5),
)
# report.status : FILLED / PARTIALLY_FILLED / REJECTED
# report.fill   : the underlying FillResult
# report.tag    : "momentum-alpha", carried through for attribution
```

The engine session that will hold a pending-order queue is not yet
shipped; in v1 `submit_order` is called directly.

## Shared kernel contract

Follows the standard [kuant kernel
contract](../README.md#shared-kernel-contract):

- Errors are `KuantValueError`, `KuantShapeError`. Every message
  names the kernel, the offending value, a stable code like
  `KE-VAL-CONTRACT`, and a one-line fix.
- Enums inherit `int` or `str` so JSON and parquet round-trip
  without custom encoders.

## Related subpackages

- [`liquidity/`](../liquidity/README.md): `execute_fill` is the
  destination of every submitted order.
- [`lifecycle/`](../lifecycle/README.md): the tradeable-window
  primitive that gates upstream of both fill and liquidity.

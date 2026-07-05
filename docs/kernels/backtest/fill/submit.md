# submit - submit_order and FillReport

## Purpose

Route an `Order` through a `LiquidityProfile` and a `FillModel` to a
`FillReport` that carries the tracking metadata a portfolio needs
to reconcile the fill back to its originating order. Translate the
categorical `reason` string from the liquidity layer's `FillResult`
into a typed `OrderStatus` on the report, so downstream code
switches on an enum rather than parsing strings.

## Public API

```python
from kuant.backtest.fill import FillReport, submit_order
```

### `FillReport`

Dataclass returned by `submit_order`. Wraps a `FillResult` with
per-order tracking metadata.

```python
FillReport(
    order_id: int,
    symbol: str,
    status: OrderStatus,
    fill: FillResult,
    tag: str = "",
)
```

Attributes:

- `order_id`: copied from the submitted `Order`.
- `symbol`: also copied. A `FillReport` is self-contained.
- `status`: `FILLED`, `PARTIALLY_FILLED`, or `REJECTED`. `PENDING`
  is only used by an order queue before submission.
- `fill`: the underlying `FillResult` from `execute_fill`. All fill
  price, size, and slippage detail lives on this attribute.
- `tag`: carried through from the `Order` for downstream reporting.

`.summary()` returns a short human-readable string that includes the
nested `fill.summary()`.

### `submit_order`

```python
submit_order(
    order: Order,
    profile: LiquidityProfile,
    price: float,
    model,
) -> FillReport
```

- `order`: must have `order_type == OrderType.MARKET` in v1.
- `profile`: the security's liquidity metadata. `profile.symbol`
  must equal `order.symbol`; a mismatch raises.
- `price`: reference price at submission (typically bar close).
- `model`: any object exposing `compute_slippage(size, adv, side) -> float`.

## Reason-to-status translation

`submit_order` maps `FillResult.reason` to `OrderStatus` via:

| `FillResult.reason` | `OrderStatus` |
| --- | --- |
| `OK` | `FILLED` |
| `CAPPED_PARTICIPATION` | `PARTIALLY_FILLED` |
| `BELOW_MIN_SIZE` | `REJECTED` |
| `NO_LIQUIDITY` | `REJECTED` |
| `MISSING_DATE` | `REJECTED` |

Any other reason value raises `KuantValueError
[KE-SUBMIT-UNKNOWN-REASON]`. The check is deliberate: if the
liquidity layer grows a new reason code without a matching entry
here, the fill layer discovers the drift immediately rather than
silently mislabelling status.

## Design decisions

### 1. Thin adapter, not smart router

`submit_order` deliberately does not implement smart-order-routing,
slicing, or scheduling. It routes exactly one `Order` through
`execute_fill` and wraps the result. The engine session that will
hold a pending-order queue, schedule LIMIT/STOP orders across bars,
and slice oversize orders across days is out of scope for v1.

The consequence: an oversize order returns `PARTIALLY_FILLED` with
the residual on `fill.size_rejected`, and the caller decides
whether to re-queue that residual onto the next bar. The engine
session will make that decision for the caller; v1 makes it
explicit.

### 2. Reason-to-status is a whitelist, not a fallthrough

The private `_status_from_reason` explicitly names each reason and
raises on unrecognised values. The alternative, defaulting unknown
reasons to `REJECTED`, would silently mask a version-skew bug
between the fill and liquidity layers: a new reason added to the
liquidity layer without a matching entry here would look like a
rejection instead of surfacing as a contract mismatch.

The stable code `KE-SUBMIT-UNKNOWN-REASON` is deliberately narrow
and unique: it only fires on the version-skew case, so it points
straight at the fix.

### 3. Symbol mismatch is a hard error

`order.symbol != profile.symbol` raises `KuantValueError
[KE-VAL-CONTRACT]`. The alternative, using `order.symbol` as the
authority and ignoring `profile.symbol`, would silently fill an
order for one name against another name's liquidity when a routing
bug in the caller crossed the wires. There is no legitimate reason
to submit an order for `XYZ` against `ACME`'s profile, so the
kernel refuses.

### 4. Non-MARKET orders raise, not queue

`submit_order` on a `LIMIT` or `STOP` order raises `KuantValueError
[KE-VAL-CONTRACT]` in v1. The engine session will provide the queue;
the fill kernel refuses to silently drop the order or convert it to
MARKET. A caller who constructs a `LIMIT` order today can hold it
in their own queue until the engine ships, but the fill layer will
not execute it.

### 5. Signed size is computed inside submit_order, not stored on Order

`submit_order` reads `order.signed_size` and passes it to
`execute_fill` as the `size` argument. The `Order` stores unsigned
`size` and a `side`; the fill kernel wants signed. Doing the
conversion at the boundary keeps the strategy-facing type
sign-clean and the fill-facing type sign-native.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `order.order_type != OrderType.MARKET` | raises `KuantValueError [KE-VAL-CONTRACT]` |
| `order.symbol != profile.symbol` | raises `KuantValueError [KE-VAL-CONTRACT]` |
| `FillResult.reason` unknown to the translator | raises `KuantValueError [KE-SUBMIT-UNKNOWN-REASON]` |
| `execute_fill` raises (bad price, bad size, model contract) | propagates unchanged |
| Fill returns `CAPPED_PARTICIPATION` | `FillReport.status = PARTIALLY_FILLED` with residual on `fill.size_rejected` |
| Fill returns any rejection reason | `FillReport.status = REJECTED`, `fill.price = NaN` |

## Examples

### End-to-end: fully filled market buy

```python
>>> import pandas as pd
>>> from datetime import date
>>> from kuant.backtest.fill import (
...     Order, OrderSide, OrderStatus, submit_order,
... )
>>> from kuant.backtest.liquidity import LiquidityProfile, FlatSlippage
>>> adv = pd.Series([1_000_000.0], index=[pd.Timestamp("2020-01-02")])
>>> profile = LiquidityProfile(
...     symbol="XYZ", adv_series=adv, min_size=100.0,
...     max_participation=0.10,
... )
>>> order = Order(
...     symbol="XYZ",
...     side=OrderSide.BUY,
...     size=1000.0,
...     timestamp=date(2020, 1, 2),
...     tag="momentum-alpha",
... )
>>> report = submit_order(
...     order, profile=profile, price=50.0, model=FlatSlippage(bps=5),
... )
>>> report.status is OrderStatus.FILLED
True
>>> report.symbol
'XYZ'
>>> report.tag
'momentum-alpha'
>>> round(report.fill.price, 4)
50.025
>>> report.fill.size_filled
1000.0
>>> report.fill.size_rejected
0.0
```

The 1000-share buy at 50.0 with 5 bps flat slippage fills at
`50.025`. `report.fill.reason == "OK"` is mapped to
`OrderStatus.FILLED` on the report. The `tag` rides through
untouched.

### Partial fill

```python
>>> big = Order(
...     symbol="XYZ",
...     side=OrderSide.BUY,
...     size=250_000.0,
...     timestamp=date(2020, 1, 2),
... )
>>> report = submit_order(
...     big, profile=profile, price=50.0, model=FlatSlippage(bps=5),
... )
>>> report.status is OrderStatus.PARTIALLY_FILLED
True
>>> report.fill.reason
'CAPPED_PARTICIPATION'
>>> report.fill.size_filled
100000.0
>>> report.fill.size_rejected
150000.0
```

The oversized order is truncated by the participation cap; the
report's status is `PARTIALLY_FILLED` and the 150,000-share residual
sits on `fill.size_rejected` for the caller to re-queue or drop.

### Rejected below min_size

```python
>>> tiny = Order(
...     symbol="XYZ",
...     side=OrderSide.BUY,
...     size=50.0,
...     timestamp=date(2020, 1, 2),
... )
>>> report = submit_order(
...     tiny, profile=profile, price=50.0, model=FlatSlippage(bps=5),
... )
>>> report.status is OrderStatus.REJECTED
True
>>> report.fill.reason
'BELOW_MIN_SIZE'
>>> import math; math.isnan(report.fill.price)
True
```

### Symbol mismatch

```python
>>> stray = Order(
...     symbol="ACME",
...     side=OrderSide.BUY,
...     size=1000.0,
...     timestamp=date(2020, 1, 2),
... )
>>> submit_order(stray, profile=profile, price=50.0,
...              model=FlatSlippage(bps=5))
Traceback (most recent call last):
    ...
kuant.errors.KuantValueError: ...
```

The `profile.symbol` is `XYZ` and the `order.symbol` is `ACME`; the
kernel refuses rather than filling against the wrong name's
liquidity.

## Cross-check tests

- For every reason in the whitelist, the returned status matches
  the table above.
- `report.order_id == order.order_id` and
  `report.symbol == order.symbol` on every returned report.
- `report.fill` is the same `FillResult` instance
  `execute_fill` returned; nothing rewraps it.
- Non-MARKET orders raise; MARKET orders always return a report.

## Related kernels

- [`order.md`](order.md): the `Order` type consumed here.
- `kuant.backtest.liquidity.execute_fill`: the destination of every
  submission; owns the `reason` codes this kernel maps.
- `kuant.backtest.liquidity.FillResult`: nested on every
  `FillReport` as the `.fill` attribute.

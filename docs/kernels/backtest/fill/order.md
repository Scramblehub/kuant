# order - Order dataclass and companion enums

## Purpose

Encode a single order intent as a typed record with an explicit side
and an unsigned quantity, so that direction and magnitude never
collapse into an ambiguous signed scalar. The enums `OrderSide`,
`OrderType`, and `OrderStatus` provide typed values for the three
categorical axes of an order's life: which way, what fill semantics,
and what happened.

## Public API

```python
from kuant.backtest.fill import Order, OrderSide, OrderType, OrderStatus
```

### `OrderSide`

Int-valued enum. Values are `+1` and `-1` so arithmetic composes.

| Value | Semantics |
| --- | --- |
| `BUY = +1` | Long-direction fill; positive `signed_size` |
| `SELL = -1` | Short-direction fill; negative `signed_size` |

### `OrderType`

String-valued enum. Only `MARKET` is executable in v1.

| Value | Semantics |
| --- | --- |
| `MARKET` | Executes at reference price plus slippage |
| `LIMIT` | Reserved; requires `limit_price` at construction |
| `STOP` | Reserved; requires `limit_price` at construction |

`LIMIT` and `STOP` are reserved on the enum so callers can construct
them and the engine layer (arriving later) can schedule them. In v1
they raise from `submit_order` with `KE-VAL-CONTRACT`.

### `OrderStatus`

String-valued enum. Terminal or in-flight state after submission.

| Value | Semantics |
| --- | --- |
| `PENDING` | In queue; not yet routed to `submit_order` |
| `FILLED` | Fully filled at the requested size |
| `PARTIALLY_FILLED` | Truncated by the ADV participation cap |
| `REJECTED` | Refused: below min size, no liquidity, or missing date |
| `CANCELED` | Withdrawn by the caller before submission |

### `Order`

Dataclass. One record per order intent.

```python
Order(
    symbol: str,
    side: OrderSide,
    size: float,                                # unsigned, > 0
    timestamp: datetime.date,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    order_id: int = <auto>,                     # monotonic default
    tag: str = "",
)
```

- `signed_size` property returns `side.value * size` for callers
  who need the signed number.
- `.summary()` returns a short human-readable string.

## Design decisions

### 1. Explicit side plus unsigned size

The single most common sign convention bug in strategy code is
"is a negative size a sell, or the previous position minus the new
target, or a short?" `Order` banishes the ambiguity by refusing to
carry sign on `size` at all: `size > 0` is enforced at construction,
and direction lives on `side`. A caller who wants the signed
quantity for arithmetic uses the `signed_size` property.

The trade-off: strategy code that historically emitted signed
scalars needs a small adapter to construct an `Order`. That is the
right price to pay for making sign conventions impossible to invert
by accident. Every subsequent kernel that consumes the order
(`submit_order`, `execute_fill`) reads `signed_size`, so the
signed-view is available where it is needed.

### 2. MARKET only in v1

`OrderType.LIMIT` and `OrderType.STOP` are reserved on the enum but
not executable. `submit_order` raises `KuantValueError
[KE-VAL-CONTRACT]` on anything other than `MARKET`. Reservation
lets callers build the objects today (for a later engine layer's
queue) without forcing the enum to grow when the engine ships.

The constructor still validates `limit_price` for reserved types: a
`LIMIT` or `STOP` order with a missing or non-positive `limit_price`
raises at construction. Fail early rather than fail at scheduling.

### 3. Auto-assigned `order_id`

`order_id` defaults to the next value from a module-level monotonic
counter. Callers who need deterministic replay (test suites,
reconciliation across runs) pass an explicit `order_id`. The counter
is process-global; two `Order`s constructed in the same process are
guaranteed distinct without any coordination from the caller.

The alternative, a UUID, buys uniqueness across processes at the
cost of readability in logs. The counter's cheap integers won on
ergonomics. Callers who need cross-process uniqueness supply their
own scheme via the constructor argument.

### 4. Free-form `tag`

`tag` is a string that no kernel consumes; it rides through
`FillReport` for reporting and attribution. Typical uses: strategy
name, cohort id, execution algorithm name, batch identifier.
Free-form because the reporting layer downstream is out of scope
for the kernels; a strategy that produces per-cohort attribution
tags for a downstream aggregator would over-constrain everyone if
the kernel imposed a schema.

### 5. `timestamp` is a `datetime.date`, not a `Timestamp`

The kernel accepts `datetime.date` on the type; `_as_date` in the
lifecycle layer normalises `date`, `pd.Timestamp`, and
`np.datetime64` to a common form on the way into `execute_fill`.
Typing on `date` reflects the calendar granularity of a bar-level
backtest: intra-bar ordering of same-day fills is a scheduling
concern for the engine, not a per-order attribute.

### 6. Constructor validates size positivity

`require_positive(size, "size", kernel="Order")` rejects zero and
negative values at construction. A zero-size order is a no-op that
should never be constructed; a negative size is a sign-convention
bug the "explicit side" design is meant to prevent. Fail at the
boundary.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `size <= 0` | raises `KuantValueError` via `require_positive` |
| `order_type = LIMIT` with `limit_price = None` | raises `KuantValueError [KE-VAL-CONTRACT]` |
| `order_type = STOP` with `limit_price = None` | raises `KuantValueError [KE-VAL-CONTRACT]` |
| `limit_price` present but non-finite or `<= 0` on LIMIT/STOP | raises `KuantValueError [KE-ORDER-LIMIT-INVALID]` |
| Explicit `order_id` collides with the auto-counter | permitted; caller owns uniqueness when overriding |
| `tag` is any string, including empty | permitted; not consumed by kernels |

## Examples

### A market buy with a tag

```python
>>> from datetime import date
>>> from kuant.backtest.fill import Order, OrderSide
>>> o = Order(
...     symbol="XYZ",
...     side=OrderSide.BUY,
...     size=1000.0,
...     timestamp=date(2020, 1, 2),
...     tag="momentum-alpha",
... )
>>> o.symbol
'XYZ'
>>> o.signed_size
1000.0
>>> o.tag
'momentum-alpha'
```

### A market sell

```python
>>> o = Order(
...     symbol="ACME",
...     side=OrderSide.SELL,
...     size=500.0,
...     timestamp=date(2020, 1, 2),
... )
>>> o.signed_size
-500.0
```

The unsigned `size` is 500; the `signed_size` property applies the
side.

### Reserved LIMIT type

```python
>>> from kuant.backtest.fill import OrderType
>>> o = Order(
...     symbol="XYZ",
...     side=OrderSide.BUY,
...     size=1000.0,
...     timestamp=date(2020, 1, 2),
...     order_type=OrderType.LIMIT,
...     limit_price=49.50,
... )
>>> o.order_type
<OrderType.LIMIT: 'limit'>
```

The Order constructs cleanly; `submit_order` will refuse it in v1
until the engine layer's LIMIT scheduling ships.

### Invalid size

```python
>>> Order(
...     symbol="XYZ",
...     side=OrderSide.BUY,
...     size=0.0,
...     timestamp=date(2020, 1, 2),
... )
Traceback (most recent call last):
    ...
kuant.errors.KuantValueError: ...
```

Zero size is refused at the boundary.

## Related kernels

- [`submit.md`](submit.md): `submit_order` consumes `Order` and
  returns a `FillReport`.
- `kuant.backtest.liquidity.execute_fill`: the eventual destination
  of `signed_size` after `submit_order` routes.

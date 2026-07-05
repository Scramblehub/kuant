# portfolio - PortfolioState and EquitySnapshot

## Purpose

Own the top-level ledger for a backtest run: a cash balance and a
`dict[symbol, Position]`. Consume `FillReport` objects atomically so
cash and the per-symbol `Position` advance together. Produce
`EquitySnapshot` records for stitching an equity curve. See
[README](README.md) for the framing.

## Public API

```python
from kuant.backtest.position import EquitySnapshot, PortfolioState
```

### `PortfolioState`

Mutable dataclass.

```python
PortfolioState(
    cash: float = 0.0,
    positions: dict = field(default_factory=dict),
)
```

- `cash`: available cash balance. Can go negative; no non-negative
  constraint is enforced here. Cash gating is an engine-level
  policy, not a portfolio-layer invariant.
- `positions`: `dict[str, Position]`. Symbols with `size == 0` are
  retained so `realized_pnl` survives across close-and-reopen
  sequences.

### Methods

- `apply_fill(report: FillReport) -> None`. Debit or credit cash,
  then advance the per-symbol `Position` in one call.
- `total_value(prices: dict[str, float]) -> float`. Cash plus the
  sum of `position.market_value(price)` across held positions.
- `mark_to_market(prices: dict[str, float]) -> EquitySnapshot`. All
  fields in one pass.
- `summary() -> str`. Human-readable snapshot for logs.

### `EquitySnapshot`

Immutable-in-intent value type returned by `mark_to_market`.

```python
EquitySnapshot(
    cash: float,
    positions_value: float,
    total_value: float,
    n_positions: int,
    unrealized_pnl: float,
    realized_pnl: float,
)
```

- `cash`: portfolio cash at snapshot time.
- `positions_value`: sum of signed `market_value(price)` across
  symbols with nonzero size.
- `total_value`: `cash + positions_value`. The number that lands on
  the equity curve.
- `n_positions`: count of symbols with nonzero size at snapshot
  time. Symbols that closed to flat but retain `realized_pnl` are
  excluded from this count.
- `unrealized_pnl`: sum of `position.unrealized_pnl(price)` across
  held positions.
- `realized_pnl`: sum of `position.realized_pnl` across the entire
  `positions` dict, including flat symbols.
- `summary() -> str`: human-readable snapshot.

## Design decisions

### 1. Atomic fill application

`apply_fill(report)` performs cash and position updates inside one
method call, in a fixed order: validate, debit cash, advance the
`Position`. There is no partial-success path. Either the entire
`FillReport` lands or an exception is raised before any state
mutation.

Rationale: the invariant `total_value == cash + sum(market_value)`
must hold at every observable point in the caller's loop. A method
that mutated cash and then failed on the position update would
leave the ledger inconsistent between the crash and the caller's
`except` block.

The debit direction is signed: `cash -= fill.size_filled *
fill.price`. A buy (`size_filled > 0`) decreases cash; a sell
(`size_filled < 0`) increases it. There is no long / short branch.

### 2. Rejected fills are no-ops

If `report.fill.size_filled == 0.0`, `apply_fill` returns
immediately without touching cash or positions. The fill layer sets
`size_filled = 0` on any rejection reason (`MISSING_DATE`,
`NO_LIQUIDITY`, `SIZE_ZERO`, and so on), so a portfolio-layer
caller can pass every `FillReport` in without pre-filtering.

The size-zero short-circuit runs before the price-validity check.
A rejected fill legitimately carries `NaN` as its `price`; that is
not an error, because there is no cash movement to compute.

### 3. Non-finite price on a NONZERO fill raises

If `size_filled != 0` and `price` is non-finite, `apply_fill`
raises `KuantValueError [KE-PORTFOLIO-FILL-PRICE-INVALID]` before
any state mutation. This can only happen if a caller hand-built a
`FillReport` or a custom liquidity model violates the contract that
`execute_fill` sets `price = NaN` only when `size_filled = 0`.

The alternative is silently zeroing cash into NaN, which would
poison every downstream `total_value` call. Fail loudly at the
insertion point.

### 4. Strict prices contract on `total_value` and `mark_to_market`

Both methods raise `KuantValueError [KE-VAL-SCHEMA]` if a symbol
with an open position is missing from the `prices` dict. Flat
symbols never enter the price lookup, so a `dict` that omits
already-closed names is fine.

The strict contract is deliberate. A missing entry is almost
certainly a caller bug (forgot to align the reference-price
dictionary with the current portfolio) rather than an intentional
signal, and the failure mode of a silent skip is a total equity
figure that understates exposure without any diagnostic. Callers
who genuinely want to accept a symbol as unpriced pass an explicit
`NaN` value; that path routes to design decision 5.

### 5. NaN prices propagate, and warn

A `NaN` price for a symbol with an open position is accepted
without raising. `market_value` and `unrealized_pnl` both produce
NaN, which propagates into `positions_value`, `total_value`, and
`unrealized_pnl` on the returned snapshot.

Simultaneously, `mark_to_market` emits a `KuantNumericWarning` with
code `KW-PORTFOLIO-NAN-MARK` that names up to the first five
affected symbols and offers three fixes: close the position at the
last known finite mark, apply the lifecycle terminal action, or
accept the NaN if the omission is intentional.

The semantic matches the lifecycle layer's convention: NaN means
"unpriced today," not "worth zero." Silent zeroing would inflate
returns on delisted names; silent skipping would understate
exposure. Propagation preserves the mathematical truth that the
snapshot is ill-defined, and the warning gives the caller the
information they need to fix the cause upstream.

### 6. Flat positions retained in the dict

`positions` never evicts a symbol that closed to flat. Two reasons:

1. `realized_pnl` on a closed symbol contributes to
   `EquitySnapshot.realized_pnl`. Dropping the record would silently
   drop realized P&L history.
2. A close-and-reopen sequence on the same symbol would otherwise
   allocate a fresh `Position` with `realized_pnl = 0`, discarding
   the earlier realized number.

The `n_positions` field on `EquitySnapshot` filters to nonzero
sizes, so retained flat entries do not inflate the reported open
count.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `report.fill.size_filled == 0.0` | no-op (rejected or SIZE_ZERO fill) |
| `report.fill.price` non-finite AND `size_filled != 0` | raises `KuantValueError [KE-PORTFOLIO-FILL-PRICE-INVALID]` |
| `total_value` called with an empty `prices` dict AND at least one open position | raises `KuantValueError [KE-VAL-SCHEMA]` |
| `mark_to_market` called with an empty `prices` dict AND at least one open position | raises `KuantValueError [KE-VAL-SCHEMA]` |
| `prices[sym]` is NaN for an open position | `KW-PORTFOLIO-NAN-MARK` warns; NaN propagates into snapshot |
| Symbol has `size == 0` but nonzero `realized_pnl` | included in `EquitySnapshot.realized_pnl`, excluded from `n_positions` |
| `mark_to_market` on an empty portfolio | returns zeros; no warning |

## Examples

### Buy, mark, sell

```python
>>> from kuant.backtest.position import PortfolioState
>>> from kuant.backtest.fill import FillReport, OrderStatus
>>> from kuant.backtest.liquidity import FillResult
>>> ps = PortfolioState(cash=100_000.0)
>>> buy = FillReport(
...     order_id=1, symbol='XYZ',
...     status=OrderStatus.FILLED,
...     fill=FillResult(
...         price=50.0, size_filled=100.0, size_rejected=0.0,
...         slippage_bps=0.0, reason='OK', cost=5000.0,
...     ),
... )
>>> ps.apply_fill(buy)
>>> ps.cash, ps.positions['XYZ'].size
(95000.0, 100.0)
>>> snap = ps.mark_to_market({'XYZ': 55.0})
>>> snap.cash, snap.positions_value, snap.total_value
(95000.0, 5500.0, 100500.0)
>>> snap.n_positions, snap.unrealized_pnl, snap.realized_pnl
(1, 500.0, 0.0)
```

Cash decreased by `100 * 50 = 5000`; the open long marked at 55
carries `100 * (55 - 50) = 500` unrealized. Total equity moved from
`100_000` to `100_500`.

### Strict prices contract

`total_value` raises when an open position lacks a price entry.

```python
>>> from kuant.errors import KuantValueError
>>> try:
...     ps.total_value({})
... except KuantValueError as e:
...     'KE-VAL-SCHEMA' in str(e)
True
```

The identical contract applies to `mark_to_market`. Callers who
want to accept a symbol as unpriced pass `NaN` explicitly rather
than omitting the entry.

### NaN mark propagates and warns

```python
>>> import math, warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter('always')
...     snap = ps.mark_to_market({'XYZ': float('nan')})
...     any('KW-PORTFOLIO-NAN-MARK' in str(w.message) for w in caught)
True
>>> math.isnan(snap.positions_value), math.isnan(snap.total_value)
(True, True)
>>> snap.cash
95000.0
```

`cash` remains finite; only the position-derived fields carry the
NaN.

### EquitySnapshot summary

```python
>>> print(snap.summary())
=== EquitySnapshot ===
cash:              95000.0000
positions_value:   nan
total_value:       nan
n_positions:       1
unrealized_pnl:    +nan
realized_pnl:      +0.0000
```

## Cross-references

- [`position.md`](position.md): the `Position` state machine that
  `PortfolioState.apply_fill` drives on each `FillReport`.
- `kuant.backtest.fill.submit_order`: the sole intended producer of
  `FillReport` objects. Its `execute_fill` sets `price = NaN` only
  on rejected (`size_filled = 0`) fills; the portfolio layer relies
  on that invariant.
- `kuant.backtest.lifecycle`: NaN price semantics match the
  lifecycle convention. On delisting, the caller applies the
  terminal action against the equity curve rather than routing a
  degenerate fill through `apply_fill`.
- [`warmup/`](../warmup/README.md): the reference-price dictionary
  `mark_to_market` consumes typically comes from the warmup layer's
  precomputed close panel.

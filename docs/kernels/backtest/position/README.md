# kuant.backtest.position

Per-symbol accounting and top-level portfolio state for a backtest run.

## The gap this closes

A fill is not a trade. A fill is a delta on cash and a delta on one
symbol's position record, and the two updates must move together or
the equity curve drifts silently. Naive engines pass a raw
`(symbol, size, price)` tuple around and mutate whichever bookkeeping
structure the loop happens to reach first; the invariant `cash +
mark(positions) == prior_equity + realized_delta` becomes hard to
audit and easy to violate.

`kuant.backtest.position` treats accounting as a two-primitive
kernel. `Position` handles per-symbol size, average cost, and
realized P&L under netting semantics. `PortfolioState` owns the cash
balance and the `dict[symbol, Position]`, and consumes `FillReport`
objects atomically so both sides of the ledger advance in lockstep.
The engine's inner loop hands each `FillReport` to
`PortfolioState.apply_fill` and never touches a `Position`
directly.

## Files

- [`position.md`](position.md): `Position` dataclass. Signed size,
  volume-weighted `avg_cost`, cumulative `realized_pnl`, and the
  four-case `apply_fill` state machine with mark-to-market helpers.
- [`portfolio.md`](portfolio.md): `PortfolioState` dataclass. Cash
  plus a `positions` dict, atomic `apply_fill(report)`,
  `total_value(prices)`, and `mark_to_market(prices) ->
  EquitySnapshot`.

## Netting semantics

One `Position` per symbol carries a single signed `size`. Positive
values are long, negative values are short, zero is flat. A buy
against an existing short position reduces the short first; enough
size flips it to a long. There is no separate long book and short
book, no per-lot tracking, and no FIFO or LIFO close ordering. Every
close consumes at the volume-weighted `avg_cost` of the currently
open portion.

Rationale for the v1 lock:

1. Backtest attribution reports do not distinguish "long lot opened
   at $50, closed at $70" from "long size 100 at avg $50, sold at
   $70." The realized P&L is the same either way. Adding a per-lot
   ledger is real cost with no analytic payoff for the strategies
   the engine targets.
2. Tax-lot accounting (HIFO, LIFO, specific-identification) is a
   post-hoc overlay on a live book, not a backtest primitive.
   Callers who need it consume the `FillReport` stream separately
   and reconstruct lots offline.
3. Netting matches how a prime broker reports net exposure on a
   single instrument. Backtests that use `PortfolioState` and live
   accounts that use a broker-reported net position produce
   comparable equity curves without a semantic bridge.

Long-short segregation may return as an opt-in mode later. It is
deliberately out of scope for v1.

## Typical caller flow

```python
from kuant.backtest.fill import submit_order
from kuant.backtest.position import PortfolioState

state = PortfolioState(cash=100_000.0)
for order in orders_this_bar:
    report = submit_order(order, market_data)
    state.apply_fill(report)
snapshot = state.mark_to_market(reference_prices)
equity_curve.append(snapshot)
```

The engine loop repeats this pattern per bar; `EquitySnapshot`
records stitch into the equity curve without any further math on the
caller's side.

## Shared kernel contract

Both dataclasses follow the standard [kuant kernel
contract](../README.md#shared-kernel-contract):

- Errors are `KuantValueError` with stable codes
  (`KE-POS-SIZE-INVALID`, `KE-POS-PRICE-INVALID`,
  `KE-PORTFOLIO-FILL-PRICE-INVALID`, `KE-VAL-SCHEMA`). Every
  message names the kernel, the offending value, and a one-line
  fix.
- Warnings ride the `KuantNumericWarning` category
  (`KW-PORTFOLIO-NAN-MARK`) so the caller can filter or promote
  them explicitly.
- Mutations are in-place and atomic within a single method call.
  No lazy proxy objects, no deferred computation.

## Related subpackages

- [`fill/`](../fill/README.md): produces the `FillReport` objects
  that `PortfolioState.apply_fill` consumes. The fill layer owns
  price discovery and rejection reasons; the position layer owns
  the ledger update.
- [`lifecycle/`](../lifecycle/README.md): supplies the tradeable
  window and terminal actions. `mark_to_market` respects NaN prices
  the way lifecycle-aware kernels do; the two layers share the
  convention that NaN means "unpriced today," not "worth zero."
- [`warmup/`](../warmup/README.md): precomputes the reference prices
  a `mark_to_market` call needs each bar.

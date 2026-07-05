# kuant.backtest.engine

Reference orchestrator that wires the correctness-first primitives
into a single bar-driven simulation loop.

## What this subpackage is

`kuant.backtest` publishes five primitives that each solve one
narrow simulation problem correctly:

- [`warmup`](../warmup/README.md): materialize a query surface over
  indicators, universe membership, lifecycle windows, and liquidity
  masks.
- [`lifecycle`](../lifecycle/README.md): first-class listing and
  delisting semantics, so the tradeable window gates orders instead
  of NaN prices.
- [`liquidity`](../liquidity/README.md): `LiquidityProfile`, ADV
  caps, and the size-aware slippage models.
- [`fill`](../fill/README.md): `Order` intents and the
  `submit_order` path that turns an intent plus a profile plus a
  reference price into a `FillReport`.
- [`position`](../position/README.md): `PortfolioState` book-keeping,
  fills applied, mark-to-market, realized and unrealized P&L.

The engine is the small assembly that runs the loop. It iterates
the price panel, calls the user's strategy on each bar, gates each
order against the cache and the profiles, submits the survivors
through the fill path, applies the reports to the portfolio state,
and marks to market at bar close.

## Why it is intentionally small

The engine module is roughly two hundred lines. It exists so a
caller with a strategy in hand can run a backtest without gluing
five kernels together by hand, and so the recommended composition
of those kernels lives in one place that the test suite can pin.

It deliberately does not offer multi-strategy composition, event
callbacks, cross-asset margining, real-time hooks, next-bar fill
semantics, or auto-liquidation on delisting. Users who need any of
those build on top of the primitives directly. The five kernels
below the engine are the stable public surface; the engine is one
opinionated way to compose them.

## Exports

```python
from kuant.backtest.engine import (
    run,
    BacktestResult,
    StrategyFn,
)
```

- `run(cache, strategy, *, liquidity_profiles, fill_model, initial_cash, lifecycles=None)`:
  the loop. Returns a `BacktestResult`.
- `BacktestResult`: dataclass snapshot of the run. Per-bar equity
  frame, per-order trades frame (including gated intents), final
  portfolio state, order counters, `summary()` and `to_parquet()`
  helpers.
- `StrategyFn`: type alias for
  `Callable[[WarmupCache, PortfolioState, pd.Timestamp], list[Order]]`.

## Files

- [`engine.md`](engine.md): full signature, gating cascade, bar
  semantics, terminal-action policy, `BacktestResult` field
  reference, and an end-to-end worked example.

## Typical caller flow

```python
from kuant.backtest.warmup import Warmup
from kuant.backtest.liquidity import LiquidityProfile, FlatSlippage
from kuant.backtest.engine import run

warmup = Warmup(prices, mode="eager")
warmup.add_lifecycles(lifecycles)
cache = warmup.materialize()

profiles = {
    "AAA": LiquidityProfile(symbol="AAA", adv_series=adv_aaa, ...),
    "BBB": LiquidityProfile(symbol="BBB", adv_series=adv_bbb, ...),
}

result = run(
    cache,
    my_strategy,
    liquidity_profiles=profiles,
    fill_model=FlatSlippage(bps=5.0),
    initial_cash=100_000.0,
)
print(result.summary())
```

`my_strategy` is any callable matching `StrategyFn`. It returns a
list of `Order` intents per bar; the engine takes care of gating,
filling, and book-keeping.

## Shared kernel contract

Every kernel in `kuant.backtest` follows the standard [kuant kernel
contract](../../README.md#shared-kernel-contract):

- Errors are `KuantValueError`, `KuantShapeError`, and
  `KuantDependencyError`. Every message names the kernel, the
  offending value, a stable error code, and a one-line fix.
- Gated orders are recorded, not silently dropped. The trades
  frame includes every intent the strategy returned, with a
  categorical `reason` field.
- Backend and dtype are preserved across the composed loop.

## Related subpackages

- [`data/`](../data/README.md): `align`, `panelize`, `stitch`. Run
  these upstream of `Warmup` to get onto a common calendar.
- [`edgecases/`](../edgecases/README.md): NaN policies for
  non-lifecycle-driven missingness (halts, one-day quote gaps).

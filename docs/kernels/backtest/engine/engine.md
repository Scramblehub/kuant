# engine - `run`, `BacktestResult`, `StrategyFn`

## Purpose

Compose `warmup`, `lifecycle`, `liquidity`, `fill`, and `position`
into a single bar-driven loop. One entry point (`run`), one result
object (`BacktestResult`), one strategy signature (`StrategyFn`).
The module is roughly two hundred lines by design; see
[README](README.md) for the framing.

## Public API

```python
from kuant.backtest.engine import (
    run,
    BacktestResult,
    StrategyFn,
)
```

### `run`

```python
run(
    cache: WarmupCache,
    strategy: StrategyFn,
    *,
    liquidity_profiles: dict[str, LiquidityProfile],
    fill_model,
    initial_cash: float,
    lifecycles: dict[str, SecurityLifecycle] | None = None,
) -> BacktestResult
```

Parameters:

- `cache`: a materialized `WarmupCache`. Its `prices` DataFrame
  drives the bar iteration; its `.tradeable`, `.liquid`, and
  `.universe` gates are queried per bar. Build one with
  `Warmup(prices, mode=...).materialize()`.
- `strategy`: any callable matching `StrategyFn`. Signature is
  `strategy(cache, portfolio_state, timestamp) -> list[Order]`.
  Return an empty list to do nothing on a bar. The state is passed
  for read access to current cash and positions; the strategy MUST
  NOT mutate it. Mutation is not defended against at runtime, and
  is the single easiest way to corrupt a run.
- `liquidity_profiles`: symbol-to-`LiquidityProfile` mapping. A
  symbol without a registered profile is gated at order time with
  reason `NO_PROFILE`. Extra profiles for symbols never traded are
  harmless.
- `fill_model`: any object exposing
  `compute_slippage(size, adv, side) -> float`. The three provided
  models are `FlatSlippage`, `LinearImpact`, and `SquareRootImpact`;
  any custom class with a matching method is accepted.
- `initial_cash`: starting cash balance, passed to
  `PortfolioState`. Must be strictly positive.
- `lifecycles`: optional. When the cache was built with
  `Warmup.add_lifecycles`, the cache already knows the tradeable
  panel and this argument stays `None`. It exists to accept a
  post-hoc lifecycle mapping without rebuilding the cache; the
  engine does not currently rebuild the tradeable panel from it,
  so the practical recommendation is to register lifecycles on the
  `Warmup` and leave this argument at its default.

Errors:

- `KuantValueError [KE-VAL-POSITIVE]` if `initial_cash` is not
  strictly positive.
- `KuantValueError [KE-VAL-CONTRACT]` if `fill_model` has no
  `compute_slippage` method.
- `KuantShapeError [KE-SHAPE-EXPECTED]` if `liquidity_profiles`,
  `lifecycles`, or `cache.prices` is the wrong shape.

### `BacktestResult`

Dataclass snapshot returned by `run`.

| Field | Type | Meaning |
| --- | --- | --- |
| `equity` | `pd.DataFrame` | per-bar snapshot, columns below |
| `trades` | `pd.DataFrame` | per-order record, including gated intents |
| `portfolio_final` | `PortfolioState` | terminal state after the last bar |
| `initial_cash` | `float` | echoed input |
| `n_bars` | `int` | number of rows in `cache.prices` |
| `n_orders_seen` | `int` | total intents returned by `strategy` |
| `n_orders_filled` | `int` | reports with nonzero `size_filled` |
| `n_orders_rejected` | `int` | reports with zero `size_filled` from the fill layer |
| `n_orders_gated` | `int` | intents skipped before the fill layer |

`equity` columns:

- `cash`: cash balance at bar close.
- `positions_value`: sum of `size * price` across open positions.
- `total_value`: `cash + positions_value`.
- `unrealized_pnl`: mark-to-market P&L across open positions.
- `realized_pnl`: cumulative realized P&L across the run.

`trades` columns: `timestamp`, `order_id`, `symbol`, `side`,
`requested_size`, `fill_price`, `size_filled`, `slippage_bps`,
`cost`, `status`, `reason`, `tag`. Every intent produces exactly
one row. `status` is `"gated"` for intents killed before the fill
layer and takes the `OrderStatus` value otherwise (`"filled"`,
`"partially_filled"`, `"rejected"`). `reason` is a categorical
string; gated reasons are enumerated in the next section.

`BacktestResult.summary()` returns a short human-readable string
with the counters and the total return.

`BacktestResult.to_parquet(path)` writes two files:
`f"{path}_equity.parquet"` and `f"{path}_trades.parquet"`. Requires
`pyarrow`. Timestamps are stringified for portable schemas.

### `StrategyFn`

```python
StrategyFn = Callable[
    [WarmupCache, PortfolioState, pd.Timestamp],
    list[Order],
]
```

The `Timestamp` is exactly the row label from `cache.prices.index`
for the current bar. Use it to slice indicators via
`cache.get(name, timestamp, symbol=...)` and to stamp the returned
`Order` objects.

## Bar semantics

Orders emitted on bar `t` fill at bar `t`'s reference price. The
reference price is `cache.prices.loc[t, symbol]`. This is the
close-to-close convention: the strategy sees the same price row
that the fill uses, and the mark-to-market at the end of the bar
uses that same row.

Strategies that need one-bar-ahead semantics must lag their signal
internally by one bar. A `mode="next_bar"` opt-in that would fill
each order at bar `t+1`'s reference price is deferred to a later
version.

Rationale: the close-to-close convention is the simplest one that
composes cleanly with the `WarmupCache` query surface. Every
kernel that materializes an indicator against the price panel
returns a value keyed on the same date the strategy sees; forcing
strategies to be explicit about their own lag keeps the engine
from having to reason about where the lookahead lives.

## Gating cascade

For every order the strategy returns, the engine walks four gates
in order. The first one to fail records the order in `trades`
with `status="gated"` and a categorical `reason`, then moves on to
the next order. No exception is raised; no order is silently
dropped.

| Order | Check | Failure reason |
| --- | --- | --- |
| 1 | `cache.tradeable(t, sym)` | `GATED_LIFECYCLE` |
| 2 | `sym in liquidity_profiles` | `NO_PROFILE` |
| 3 | `sym in cache.prices.columns` | `SYMBOL_NOT_IN_PANEL` |
| 4 | `np.isfinite(ref_price) and ref_price > 0` | `NO_PRICE` |

Only after all four pass does the engine call
`submit_order(order, profile, price=ref_price, model=fill_model)`
and hand the resulting report to `state.apply_fill(report)`.

The rationale for recording every gated intent (rather than
dropping them) is symmetric with the lifecycle module's rationale
for gating on the tradeable window rather than NaN: the caller
needs to be able to reconcile every intent the strategy emitted,
and downstream diagnostics need a categorical field to group on.
See [`../lifecycle/security.md`](../lifecycle/security.md) for the
adjacent argument at the panel level.

Fill-layer outcomes (`BELOW_MIN_SIZE`, `NO_LIQUIDITY`,
`MISSING_DATE`, `CAPPED_PARTICIPATION`) show up on the trade row's
`reason` field just like the gate reasons above; the distinction
is that they came from `submit_order`, not from the engine's own
gates. The counters split them: gate misses land in
`n_orders_gated`, fill-layer misses land in `n_orders_rejected`.

## Terminal actions

The engine does NOT auto-close positions on a symbol's
`delisting_date`. A strategy that holds through delisting will:

- see `cache.tradeable(t, sym)` return False on and after the
  delisting date, so it cannot emit new orders on the name;
- retain the position on `PortfolioState`;
- mark to NaN starting on the first row whose reference price is
  NaN, which is typically the row after the last live close.

The terminal return injection semantics documented in
[`../lifecycle/security.md`](../lifecycle/security.md) live on the
`LifecyclePanelResult` frames, not on the engine's equity curve.
The engine's mark-to-market uses the raw `cache.prices` row, so a
`MARK_TO_ZERO` position that was not sold before delisting shows
up as `NaN` in `positions_value` from the delisting-plus-one row
onward, which then propagates into `total_value`.

A `terminal_actions=True` opt-in that would auto-liquidate held
positions per `SecurityLifecycle.terminal_action` is deferred to a
later version. Until it ships, the intended pattern is: consume
`cache.tradeable(t, sym)` inside the strategy and emit a closing
`Order` on the last tradeable bar before delisting.

## Strategy contract

The strategy callable is invoked once per row of `cache.prices`,
in order. Its signature is fixed:

```python
def strategy(
    cache: WarmupCache,
    portfolio_state: PortfolioState,
    timestamp: pd.Timestamp,
) -> list[Order]:
    ...
```

Contract:

- Read-only access to `cache` and `portfolio_state`. Mutating
  either from inside the strategy corrupts the run and is not
  defended against at runtime.
- Return a fresh `list[Order]` each call. Returning `None` is
  tolerated and treated as an empty list.
- Each `Order` must set `symbol`, `side`, `size`, and `timestamp`.
  Use the `timestamp` argument the engine passed in; the fill path
  does not consult the order's timestamp against the current bar.
- `size` is strictly positive; direction lives on `side`. Sign-
  flip bugs from a signed size field are not possible.
- Free-form `tag` is carried through to the trades frame for later
  attribution.

## End-to-end example

A buy-and-hold strategy on a two-symbol synthetic panel.

```python
>>> import numpy as np
>>> import pandas as pd
>>> from kuant.backtest.warmup import Warmup
>>> from kuant.backtest.liquidity import LiquidityProfile, FlatSlippage
>>> from kuant.backtest.fill.order import Order, OrderSide
>>> from kuant.backtest.engine import run
>>>
>>> # Synthetic panel: two symbols, ten daily bars.
>>> idx = pd.date_range("2024-01-01", periods=10, freq="D")
>>> prices = pd.DataFrame(
...     {
...         "AAA": np.linspace(100.0, 110.0, 10),
...         "BBB": np.linspace(50.0, 55.0, 10),
...     },
...     index=idx,
... )
>>> adv_aaa = pd.Series(1_000_000.0, index=idx)
>>> adv_bbb = pd.Series(500_000.0, index=idx)
>>> profiles = {
...     "AAA": LiquidityProfile(
...         symbol="AAA", adv_series=adv_aaa,
...         min_size=1.0, max_participation=0.10,
...     ),
...     "BBB": LiquidityProfile(
...         symbol="BBB", adv_series=adv_bbb,
...         min_size=1.0, max_participation=0.10,
...     ),
... }
>>>
>>> # Warmup with no indicators; the strategy just needs the price panel.
>>> cache = Warmup(prices, mode="eager").materialize()
>>>
>>> # Buy 100 shares of each name on the first bar, then do nothing.
>>> def buy_and_hold(cache, state, timestamp):
...     if timestamp != cache.prices.index[0]:
...         return []
...     return [
...         Order(symbol="AAA", side=OrderSide.BUY, size=100.0,
...               timestamp=timestamp.date()),
...         Order(symbol="BBB", side=OrderSide.BUY, size=100.0,
...               timestamp=timestamp.date()),
...     ]
>>>
>>> result = run(
...     cache,
...     buy_and_hold,
...     liquidity_profiles=profiles,
...     fill_model=FlatSlippage(bps=5.0),
...     initial_cash=100_000.0,
... )
>>> print(result.summary())  # doctest: +ELLIPSIS
=== BacktestResult ===
bars:            10
orders seen:     2
filled:          2
rejected:        0
gated:           0
initial cash:    100000.0000
final total:     ...
total return:    ...
>>> result.equity.columns.tolist()
['cash', 'positions_value', 'total_value', 'unrealized_pnl', 'realized_pnl']
>>> result.trades["status"].tolist()
['filled', 'filled']
```

Both intents pass all four gates: the cache has no lifecycle
registered so `tradeable` returns True by fall-through; the
profile mapping covers both symbols; both are columns of the price
panel; both reference prices are finite and positive. The fills
land on the first bar; the remaining nine bars just mark to
market.

Adding a third order on a symbol with no profile would flow
through the gating cascade to `NO_PROFILE`, land in
`n_orders_gated`, and show up in `result.trades` with
`status="gated"` and `reason="NO_PROFILE"` so the caller can
reconcile the intent.

## Edge cases

| Condition | Behavior |
| --- | --- |
| Strategy returns `None` | treated as empty list; no orders processed |
| Strategy returns `[]` on every bar | run completes; equity is flat at `initial_cash`; no trade rows |
| Symbol has a profile but no price column | order gated with `SYMBOL_NOT_IN_PANEL` |
| Reference price is NaN on the bar | order gated with `NO_PRICE` |
| Reference price is zero or negative | order gated with `NO_PRICE` |
| Symbol is post-delisting per registered lifecycle | order gated with `GATED_LIFECYCLE` |
| `initial_cash <= 0` | raises `KuantValueError [KE-VAL-POSITIVE]` |
| `fill_model` lacks `compute_slippage` | raises `KuantValueError [KE-VAL-CONTRACT]` |
| `liquidity_profiles` is not a dict | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| `lifecycles` is not a dict or None | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| `cache.prices` is not a DataFrame | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| Held position with NaN price on some later bar | that bar's `positions_value` and `total_value` are NaN |

## Cross-check tests

- Every intent returned by the strategy appears exactly once in
  `result.trades`; `n_orders_seen` equals `len(result.trades)`.
- `n_orders_filled + n_orders_rejected + n_orders_gated` equals
  `n_orders_seen`.
- The `equity` frame has exactly `n_bars` rows, indexed by
  `cache.prices.index`.
- With a no-op strategy, `equity["total_value"]` is constant at
  `initial_cash` and `trades` is empty.
- With a single filled buy at bar zero and a flat-price panel,
  `total_value` at the terminal bar equals
  `initial_cash - abs(slippage_cost)`; the residual is entirely
  the slippage haircut and settles cleanly against the
  `FlatSlippage` bps parameter.

## Related kernels

- [`../warmup/README.md`](../warmup/README.md): building the cache
  the engine consumes.
- [`../lifecycle/security.md`](../lifecycle/security.md): the
  `SecurityLifecycle` records the tradeable gate reads from, and
  the framing for terminal-day accounting.
- [`../liquidity/README.md`](../liquidity/README.md):
  `LiquidityProfile`, `FlatSlippage`, `LinearImpact`,
  `SquareRootImpact`.
- [`../fill/README.md`](../fill/README.md): `Order`, `OrderSide`,
  `submit_order`, `FillReport`.
- [`../position/README.md`](../position/README.md):
  `PortfolioState`, `apply_fill`, `mark_to_market`.

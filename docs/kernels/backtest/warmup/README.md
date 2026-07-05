# kuant.backtest.warmup

The vectorized precompute layer for a backtest run.

## The gap this closes

An honest backtest inner loop is path-dependent: positions, cash,
order-book state, and rejection reasons all evolve bar by bar and
cannot be vectorized without lookahead. Everything else in a typical
run is not path-dependent. A twenty-day rolling mean over a decade
of daily closes has the same value at every bar regardless of which
orders the strategy has submitted. A tradeable-window boolean and a
liquidity-gate boolean over the same panel likewise never depend on
the path.

Naive engines pay for the vectorizable part inside the inner loop.
On every bar they recompute the same rolling window, re-check the
same tradeable mask, and re-query the same universe membership. The
per-bar cost is small, the per-run cost dominates.

`kuant.backtest.warmup` factors the vectorizable work out into a
one-shot precompute pass, then hands the engine a uniform query
surface that hides which values were cached and which are being
computed live. The engine loop only sees per-bar `.get(name, ts,
sym)` and per-bar gate checks.

## Files

- [`warmup.md`](warmup.md): `Warmup(prices, mode)`. Registration
  surface for indicators, universe membership, lifecycles, and
  liquidity profiles. `.materialize()` returns a queryable
  `WarmupCache`.
- [`cache.md`](cache.md): `WarmupCache`. Uniform `get`,
  `tradeable`, `liquid`, `universe` query methods plus
  `is_cached`, `summary`, and `materialization_time_s`
  reporting.

## The three modes

`WarmupMode` picks the memory-versus-speed tradeoff for a run.

- `EAGER` (default). Materialize every registered cache-flagged
  indicator up front against the full price panel. Highest memory
  footprint, fastest per-bar access. Right for offline research
  where the setup cost amortizes across a long run.
- `LAZY`. Cache on first access. An indicator that is registered
  but never queried by the strategy code never gets computed. Right
  for exploratory notebooks and partial backtests where the
  strategy might not touch every registered signal on every run.
- `OFF`. Never cache. Every `.get()` call recomputes the underlying
  kernel against the current panel. Right for live-trading loops
  that receive fresh data each bar, small-window backtests where
  materialization cost exceeds inner-loop savings, and iterative
  debugging where a cached stale value would mask a bug.

The mode is a per-`Warmup` setting. Every indicator registered
against a given `Warmup` starts from the mode's default.

## Per-indicator overrides

`add_indicator(..., cache=True | False | None)` overrides the
mode's default for one indicator without changing global mode.
Rationale: a research setup often mixes slow-moving signals (a
252-day rolling volatility over the full universe, worth caching
even in `OFF` mode) with fast-moving live inputs (a same-bar volume
tick that must not be cached even in `EAGER` mode). Rather than
forcing the caller to split into two `Warmup` instances, the
override lets one instance carry both.

Resolution rules:

- `cache=True`: always cached, regardless of `mode`. Materializes
  eagerly in `EAGER` mode, on first access in `LAZY`, and up front
  in `OFF` (the explicit override wins over the mode default).
- `cache=False`: never cached, regardless of `mode`. Every
  `.get()` recomputes.
- `cache=None` (default): follow the mode.

## Typical caller flow

```python
from kuant.backtest.warmup import Warmup
from kuant.stats import rollmean, rollstd

warmup = Warmup(prices, mode='eager')
warmup.add_indicator('sma20', rollmean, per_symbol=True, window=20)
warmup.add_indicator('vol20', rollstd, per_symbol=True, window=20)
warmup.add_universe_membership(membership_panel)
warmup.add_lifecycles(lifecycle_map)
warmup.add_liquidity_profiles(liquidity_map)
cache = warmup.materialize()

# engine inner loop
for ts in prices.index:
    for sym in cache.prices.columns:
        if not (cache.universe(ts, sym) and
                cache.tradeable(ts, sym) and
                cache.liquid(ts, sym)):
            continue
        signal = cache.get('sma20', ts, sym)
        # submit orders...
```

The engine's per-bar code never branches on `mode`.

## Shared kernel contract

Both types follow the standard [kuant kernel
contract](../README.md#shared-kernel-contract):

- Errors are `KuantValueError` and `KuantShapeError` with stable
  codes (`KE-VAL-RANGE`, `KE-VAL-TYPE`, `KE-VAL-DUPLICATE`,
  `KE-VAL-MISSING`, `KE-SHAPE-EXPECTED`, `KE-SHAPE-EQUAL-LEN`,
  `KE-WARMUP-EMPTY-PANEL`, `KE-WARMUP-INDICATOR-FAILED`).
- Warnings ride the `KuantNumericWarning` category
  (`KW-CACHE-TS-NOT-IN-PANEL`, `KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL`)
  so silent gate falsehoods are auditable.
- Materialization failures name the failing indicator and offer a
  standalone reproduction hook.

## Related subpackages

- [`lifecycle/`](../lifecycle/README.md): `SecurityLifecycle` and
  `tradeable_mask` supply the tradeable-panel input.
  `Warmup.add_lifecycles` maps symbols to lifecycles; the cache
  exposes `tradeable(ts, sym)`.
- `kuant.backtest.liquidity`: `LiquidityProfile` and
  `liquidity_mask` supply the liquid-panel input.
- [`position/`](../position/README.md): the engine consumes cache
  queries and hands the resulting fills into `PortfolioState`.
- `kuant.stats`: rolling kernels (`rollmean`, `rollstd`, and so on)
  are the most common `add_indicator` callables.

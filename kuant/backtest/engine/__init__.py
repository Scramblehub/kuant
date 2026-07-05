"""kuant.backtest.engine: reference orchestrator over the primitives.

Assembles `warmup`, `lifecycle`, `liquidity`, `fill`, and `position`
into a bar-driven simulation loop. Intentionally small (~200 lines).
Users who want fancier features build on top of the primitives
directly; this engine is a reference orchestrator, not a framework.

Primitives:

- `run(cache, strategy, liquidity_profiles, fill_model, initial_cash,
  lifecycles=None)`: bar iteration + gate + submit + apply + mark.
- `BacktestResult`: per-bar equity DataFrame, per-order trades
  DataFrame, final PortfolioState, order counters, `.to_parquet(path)`
  for reporting.
- `StrategyFn`: alias for
  `Callable[[WarmupCache, PortfolioState, Timestamp], list[Order]]`.

Design: docs/kernels/backtest/engine/README.md.
"""

from kuant.backtest.engine.engine import BacktestResult, StrategyFn, run

__all__ = ["BacktestResult", "StrategyFn", "run"]

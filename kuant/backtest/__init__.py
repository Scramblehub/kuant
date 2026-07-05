"""kuant.backtest: correctness-first primitives for backtest engines.

Umbrella for the pieces that make a backtest give the right answer
before it gives a fast one. Each submodule is a self-contained
primitive; the `engine` orchestrator (still to ship) composes them
into a reference simulation loop.

Submodules:

- `lifecycle`: SecurityLifecycle + TerminalAction + apply_lifecycle +
  tradeable_mask + lifecycle_returns + detect_delistings. Closes the
  silent-corruption gap around listings and delistings.
- `liquidity` (planned): LiquidityProfile + FillModel + execute_fill.
  Volume-aware fill logic that respects ADV, spread, and impact.
- `fill` (planned): Order, FillResult, order-book state.
- `position` (planned): Position accounting, cash, mark-to-market.
- `warmup` (planned): vectorized precompute layer for indicators and
  universe membership so the engine loop stays thin.
- `engine` (planned): reference orchestrator over the above.

Import each primitive from its submodule; the umbrella deliberately
does not flatten the namespace.
"""

from kuant.backtest import lifecycle

__all__ = ["lifecycle"]

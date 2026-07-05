"""kuant.backtest.warmup: precompute the vectorizable parts of a backtest.

Registration surface for indicators, universe membership, lifecycle
maps, and liquidity profiles. Materializes the eagerly-cached items
into a `WarmupCache` that the engine's inner loop queries uniformly.

Three modes trade memory vs speed:

- `EAGER` (default): materialize everything upfront. Right for offline
  backtests where setup cost amortizes across many bars.
- `LAZY`: cache-on-first-access. Right for exploratory runs.
- `OFF`: never cache. Right for live-trading loops or debugging.

Per-indicator `cache=True|False|None` override lets a caller mix cached
slow-moving indicators with live fast-moving ones.

Primitives:

- `Warmup(prices, mode)`: registration surface.
  - `add_indicator(name, kernel, per_symbol=False, cache=None, **kwargs)`
  - `add_universe_membership(membership_df)`
  - `add_lifecycles(dict[symbol -> SecurityLifecycle])`
  - `add_liquidity_profiles(dict[symbol -> LiquidityProfile])`
  - `materialize() -> WarmupCache`
- `WarmupCache`: uniform query surface.
  - `get(name, timestamp, symbol=None)`
  - `tradeable(timestamp, symbol)`
  - `liquid(timestamp, symbol)`
  - `universe(timestamp, symbol)`
  - `is_cached(name)`, `summary()`, `materialization_time_s`
- `WarmupMode`: enum EAGER, LAZY, OFF.

Design: docs/kernels/backtest/warmup/README.md.
"""

from kuant.backtest.warmup.cache import WarmupCache, WarmupMode
from kuant.backtest.warmup.warmup import Warmup

__all__ = ["Warmup", "WarmupCache", "WarmupMode"]

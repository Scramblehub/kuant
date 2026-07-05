"""kuant.backtest.position: portfolio state and mark-to-market.

Consumes `FillReport` objects from `kuant.backtest.fill.submit_order`
and updates cash + per-symbol positions. Produces `EquitySnapshot`
records that the engine (arriving next) will stitch into an equity
curve.

Primitives:

- `Position`: signed size, volume-weighted `avg_cost`, cumulative
  `realized_pnl`. Netting semantics (single signed size per symbol).
- `PortfolioState`: cash plus symbol-to-Position dict. Applies fills
  atomically (cash + position updated together), computes total value
  and marks to market.
- `EquitySnapshot`: dataclass returned by `mark_to_market`, exposing
  cash, positions_value, total_value, n_positions, unrealized_pnl,
  and realized_pnl.

Design: docs/kernels/backtest/position/README.md.
"""

from kuant.backtest.position.portfolio import EquitySnapshot, PortfolioState
from kuant.backtest.position.position import Position

__all__ = ["EquitySnapshot", "PortfolioState", "Position"]

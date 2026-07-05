"""kuant.portfolio — P&L, risk-adjusted returns, and contribution analysis.

Full-history scalar counterparts to the rolling versions in
`kuant.stats` (`rollsharpe`, `rollsortino`, `rollmdd`), plus a
two-dimensional P&L attribution kernel.

- `drawdown(equity)`: peak-to-trough series plus max, peak, trough,
  duration, and a recovered flag.
- `sharperatio(returns, ann_factor, rf)`: annualized full-history Sharpe.
- `sortinoratio(returns, ann_factor, target)`: annualized Sortino using
  downside deviation instead of full std.
- `contribution(positions, returns, group, asset_names)`: per-asset
  and per-group P&L attribution.

For rolling versions of Sharpe / Sortino / MDD see `kuant.stats`.
"""

from kuant.portfolio.contribution import ContributionResult, contribution
from kuant.portfolio.drawdown import DrawdownResult, drawdown
from kuant.portfolio.riskmetrics import (
    CaptureResult,
    DrawdownTableResult,
    UlcerResult,
    deflated_sharpe,
    down_capture,
    drawdown_table,
    kelly,
    omega,
    probabilistic_sharpe,
    ulcer_index,
    up_capture,
)
from kuant.portfolio.sharperatio import SharpeResult, sharperatio
from kuant.portfolio.sortinoratio import SortinoResult, sortinoratio

__all__ = [
    "CaptureResult",
    "ContributionResult",
    "DrawdownResult",
    "DrawdownTableResult",
    "SharpeResult",
    "SortinoResult",
    "UlcerResult",
    "contribution",
    "deflated_sharpe",
    "down_capture",
    "drawdown",
    "drawdown_table",
    "kelly",
    "omega",
    "probabilistic_sharpe",
    "sharperatio",
    "sortinoratio",
    "ulcer_index",
    "up_capture",
]

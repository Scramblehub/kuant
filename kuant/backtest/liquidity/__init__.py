"""kuant.backtest.liquidity: volume-aware fills and participation caps.

Companion to `kuant.backtest.lifecycle`. Where lifecycle asks
"does this security exist right now?", liquidity asks "can this order
fill at the stated price, and if so, how much slippage?"

Primitives:

- `LiquidityProfile`: per-security ADV, spread, min_size,
  max_participation.
- `FillModel` family: `FlatSlippage` (constant bps), `LinearImpact`
  (linear in participation rate), `SquareRootImpact` (Almgren-Chriss).
- `execute_fill`: single-order attempt returning a `FillResult` with
  fill price, size filled, size rejected, applied slippage, and a
  categorical reason.
- `execute_fill_panel`: vectorized batch fill over a DataFrame of
  orders against one profile.
- `liquidity_mask`: boolean per-date gate; True where ADV meets a
  minimum threshold. Compose with `tradeable_mask` from lifecycle for
  the full "can trade today" gate.

Design: docs/kernels/backtest/liquidity/README.md.
"""

from kuant.backtest.liquidity.execute import (
    FillResult,
    execute_fill,
    execute_fill_panel,
    liquidity_mask,
)
from kuant.backtest.liquidity.models import (
    FlatSlippage,
    LinearImpact,
    SquareRootImpact,
)
from kuant.backtest.liquidity.profile import LiquidityProfile

__all__ = [
    "FillResult",
    "FlatSlippage",
    "LinearImpact",
    "LiquidityProfile",
    "SquareRootImpact",
    "execute_fill",
    "execute_fill_panel",
    "liquidity_mask",
]

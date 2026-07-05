"""kuant.backtest.lifecycle — first-class listing / delisting semantics.

The bug this closes: most backtest engines silently ignore orders on
NaN prices and either drop the position or forward-fill the last live
price forever. On a real point-in-time equity book, both behaviors
quietly corrupt returns.

A `SecurityLifecycle` records listing_date, delisting_date, and a
terminal_action in {LIQUIDATE_AT_LAST, MARK_TO_ZERO, PRORATE_RECOVERY}
matching typical delisting-return code schemas used by exchanges and
vendor databases. Kernels then mask prices to the tradeable window,
bake the terminal transition into a return series, or produce a
boolean fill-gate a simulator can consult.

Kernels shipped:

- `SecurityLifecycle`, `TerminalAction`: the primitive and its enum.
- `apply_lifecycle` (single-symbol Series) / `apply_lifecycle_panel`
  (DataFrame): mask prices to the tradeable window.
- `lifecycle_returns`: returns with the terminal transition baked in.
- `tradeable_mask`: True/False per date. Simulators should gate on
  THIS instead of relying on NaN semantics.
- `lifecycle_panel_report`: one-call bundle of the above for a panel.
- `detect_delistings`, `lifecycles_from_panel`: heuristic fallback for
  callers without a real lifecycle table (yfinance / alpaca panels).

Design: docs/kernels/backtest/lifecycle/security.md.
"""

from kuant.backtest.lifecycle.detect import detect_delistings, lifecycles_from_panel
from kuant.backtest.lifecycle.security import (
    LifecyclePanelResult,
    SecurityLifecycle,
    TerminalAction,
    apply_lifecycle,
    apply_lifecycle_panel,
    lifecycle_panel_report,
    lifecycle_returns,
    tradeable_mask,
)

__all__ = [
    "LifecyclePanelResult",
    "SecurityLifecycle",
    "TerminalAction",
    "apply_lifecycle",
    "apply_lifecycle_panel",
    "detect_delistings",
    "lifecycle_panel_report",
    "lifecycle_returns",
    "lifecycles_from_panel",
    "tradeable_mask",
]

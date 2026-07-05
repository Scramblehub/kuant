"""kuant.edgecases — strategies for the tricky edges of real data.

Callable strategy objects for the failure modes every backtest has to
handle: NaN inputs, delisted names, outliers. Ship them once so users
compose behaviour instead of re-writing it per project.

Modules:

- `nanpolicies` — five NaN-handling strategies: `strict`, `skipna`,
  `forwardfill`, `interpolate`, `dropcolumn`. Accessible as
  `nanpolicies.forwardfill(x)` or by name via `nanpolicies.get('forwardfill')`.
- `delistedhandling` — three utilities for historical delisted names:
  `zero_after_delist`, `hold_last_price` (warns on phantom-equity spans),
  and `full_recovery_check` (survivor-bias diagnostic).
- `outlierpolicy` — single function with pluggable method:
  `'mad' | 'iqr' | 'zscore'`. Returns a boolean mask.
"""

from kuant.edgecases import nanpolicies
from kuant.edgecases.delistedhandling import (
    RecoveryCheckResult,
    full_recovery_check,
    hold_last_price,
    zero_after_delist,
)
from kuant.edgecases.outlierpolicy import outlierpolicy

__all__ = [
    "RecoveryCheckResult",
    "full_recovery_check",
    "hold_last_price",
    "nanpolicies",
    "outlierpolicy",
    "zero_after_delist",
]

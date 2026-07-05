"""Deprecated re-export path.

`kuant.lifecycle` has moved to `kuant.backtest.lifecycle` as part of the
0.4.0 reorganization that groups backtest-correctness primitives
(lifecycle, liquidity, fill, position, warmup, engine) under a single
umbrella subpackage. This module remains callable through 0.4.x so
existing imports continue to work; it will be removed in 0.5.0.

Migration (mechanical rename):

    # old (0.3.x)
    from kuant.lifecycle import SecurityLifecycle, apply_lifecycle
    from kuant.lifecycle.detect import detect_delistings

    # new (0.4.x and later)
    from kuant.backtest.lifecycle import SecurityLifecycle, apply_lifecycle
    from kuant.backtest.lifecycle.detect import detect_delistings

Importing anything from this shim raises a `KuantDeprecationWarning`
once per Python session. To promote to an error while you migrate:

    import warnings
    from kuant.errors import KuantDeprecationWarning
    warnings.filterwarnings("error", category=KuantDeprecationWarning)
"""

import warnings as _warnings

from kuant.errors import KuantDeprecationWarning as _KuantDeprecationWarning

_warnings.warn(
    "kuant.lifecycle has moved to kuant.backtest.lifecycle in v0.4.0 and "
    "will be removed in v0.5.0.  [KW-DEPRECATION-MOVE]\n"
    "  → Fix: rewrite `from kuant.lifecycle import X` as "
    "`from kuant.backtest.lifecycle import X`",
    category=_KuantDeprecationWarning,
    stacklevel=2,
)

from kuant.backtest.lifecycle import (  # noqa: E402
    LifecyclePanelResult,
    SecurityLifecycle,
    TerminalAction,
    apply_lifecycle,
    apply_lifecycle_panel,
    detect_delistings,
    lifecycle_panel_report,
    lifecycle_returns,
    lifecycles_from_panel,
    tradeable_mask,
)

# Submodule shims so `from kuant.lifecycle.security import X` and
# `from kuant.lifecycle.detect import X` continue to resolve.
from kuant.backtest.lifecycle import detect as detect  # noqa: E402, F401
from kuant.backtest.lifecycle import security as security  # noqa: E402, F401

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

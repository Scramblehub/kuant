# kuant.backtest.lifecycle

First-class listing and delisting semantics for point-in-time equity
panels.

## The gap this closes

Most backtest engines treat a security's price series as the sole
source of truth for whether the position is tradeable. When a name
delists, the series becomes NaN. Common backtest tooling then does
one of two things, both wrong:

(a) It silently ignores the order that hits a NaN price, leaving the
    position stranded on the book indefinitely.
(b) It forward-fills the last live print forever, silently inflating
    total return on a name that is worth zero.

On a survivorship-adjusted panel, either behavior can flip the sign
of long-short Sharpe. The corruption is invisible from the reports:
the engine produces a positive PnL contribution from a dead name.

`kuant.backtest.lifecycle` treats the tradeable window as a first-class
primitive. A `SecurityLifecycle` records the listing date, the
delisting date, the terminal action, and the recovery fraction for a
held position. Every downstream kernel gates on the lifecycle rather
than on NaN.

## Files

- [`security.md`](security.md): `SecurityLifecycle`,
  `TerminalAction`, `apply_lifecycle`, `apply_lifecycle_panel`,
  `lifecycle_returns`, `tradeable_mask`, `LifecyclePanelResult`,
  `lifecycle_panel_report`.
- [`detect.md`](detect.md): `detect_delistings` heuristic and
  `lifecycles_from_panel` convenience wrapper for panels without a
  real delisting table.

## The three terminal actions

`TerminalAction` enumerates what happens to a position held on the
delisting date. Values match the distinctions that typical
delisting-return code schemas used by exchanges and vendor
databases encode:

- `LIQUIDATE_AT_LAST`: sold at close on the delisting date. Terminal
  day plus one return is `0.0`. Optimistic; assumes a fill exists at
  the reported last price.
- `MARK_TO_ZERO`: bankruptcy or worthless close. Terminal day plus
  one return is `-1.0`.
- `PRORATE_RECOVERY`: reorganization, cash-plus-stock merger, or
  forced conversion with a known recovery ratio `r` in `[0, 1]`.
  Terminal day plus one return is `r - 1.0`.

## Typical caller flow

```python
from datetime import date
from kuant.backtest.lifecycle import (
    SecurityLifecycle,
    TerminalAction,
    lifecycle_panel_report,
)

lifecycles = {
    "AAA": SecurityLifecycle(symbol="AAA"),
    "BBB": SecurityLifecycle(
        symbol="BBB",
        delisting_date=date(2022, 6, 15),
        terminal_action=TerminalAction.MARK_TO_ZERO,
    ),
}
report = lifecycle_panel_report(prices, lifecycles)
# report.cleaned          : prices with post-delisting rows nulled
# report.tradeable        : boolean panel; True where an order could fill
# report.terminal_returns : injected terminal-day-plus-one returns
```

For panels without a delisting table, `detect_delistings` reads the
panel and infers dead columns from the trailing-NaN pattern.

## Shared kernel contract

Every kernel here follows the standard [kuant kernel
contract](../README.md#shared-kernel-contract):

- Backend preserved (numpy in, numpy out).
- Dtype preserved (float32 stays float32; ints promote to float64).
- Errors are `KuantValueError`, `KuantShapeError`,
  `KuantDependencyError`. Every message names the kernel, the
  offending value, a stable code, and a one-line fix.

## Related subpackages

- [`data/`](../data/README.md): `align`, `panelize`, `stitch`. These
  compose upstream of `lifecycle`. Align first, then apply lifecycle
  masks on the aligned panel.
- [`edgecases/`](../edgecases/README.md): NaN policies for
  non-lifecycle-driven missingness (halts, one-day quote gaps).

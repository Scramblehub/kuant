# detect - Heuristic delisting inference

## Purpose

Flag panel columns that have gone permanently NaN and wrap them in
`SecurityLifecycle` records, for use when no exchange or vendor
delisting table is at hand. Enables the full lifecycle machinery on
scraped, yfinance-only, or single-broker panels.

Not a substitute for a real delisting-return code table. The
heuristic infers the date of the last live print but not the reason
for the delisting; a single defaulted `terminal_action` must be
assumed across every detected symbol.

## Public API

```python
from kuant.backtest.lifecycle import detect_delistings, lifecycles_from_panel
```

- `detect_delistings(prices: pd.DataFrame, min_gap_days: int = 5) -> dict[str, date]`
- `lifecycles_from_panel(prices, min_gap_days=5, terminal_action=TerminalAction.MARK_TO_ZERO) -> dict[str, SecurityLifecycle]`

## Design decisions

### 1. The heuristic

A column is flagged as delisted at date `d` when all three hold:

1. `d` is the last non-NaN row of that column.
2. The column has at least `min_gap_days` NaN rows strictly after
   `d`.
3. The panel extends beyond `d + min_gap_days`; equivalently,
   condition 2 is achievable.

Columns still printing on the last panel row are never flagged.

Concretely, the implementation reads:

```python
last_idx = int(np.flatnonzero(finite.to_numpy())[-1])
trailing_nan = n - 1 - last_idx
if trailing_nan >= int(min_gap_days):
    out[str(col)] = _as_date(prices.index[last_idx])
```

### 2. Why `min_gap_days`, default 5

Panels have holes for reasons other than delisting: weekends,
holidays, exchange half-days, single-day quote gaps, multi-day
halts. Without a gap floor the detector would flip on every Friday
close of a raw calendar-day panel.

`min_gap_days=5` is a compromise. It exceeds a normal
Friday-to-Monday weekend and typical single-holiday runs, while
still catching real delistings within about a week of the event.

Guidance:

- Business-day calendars (weekends already dropped): `3` is safe.
- Raw calendar-day indices: keep `5` or higher.
- Panels with historically long halts (small-cap venues, foreign
  ADRs on holidays): raise to `10` or more.

The parameter is validated as a positive int; zero or negative
values raise via `require_positive`.

### 3. `lifecycles_from_panel` assumes one terminal action for all

`detect_delistings` returns only the dates. `lifecycles_from_panel`
wraps them into `SecurityLifecycle` records with the same
`terminal_action` for every detected symbol (default
`MARK_TO_ZERO`). This is the honest default: without a delisting
reason code, treating every dead name as worthless is conservative
for a long book and aggressive for a short book.

Callers with per-symbol outcome data (recovery ratios from vendor
delisting-return code tables, hand-labeled reorg events, or merger
close prices) should build the mapping directly rather than call
this wrapper.

### 4. Returns a dict, not a DataFrame

Consistency with `apply_lifecycle_panel` and
`lifecycle_panel_report`, both of which accept a
`Mapping[str, ...]`. The output of `detect_delistings` composes
with a hand-labeled mapping via `dict.update` when part of the
panel has known metadata.

### 5. All-NaN columns are not flagged

If `finite.any()` is False the column has no last real print to
anchor on. Returning a date would be meaningless. The kernel
silently skips such columns; upstream data-loading QA should catch
them.

## Edge cases

| Condition | Behavior |
| --- | --- |
| Column entirely NaN | not flagged; no anchor date exists |
| Column has trailing gap `< min_gap_days` | not flagged (looks like a halt) |
| Column has no trailing NaN | not flagged (still printing) |
| Column with a middle-of-panel gap | flagged only if the tail is also long enough NaN |
| Panel with only one row | flagged only if `min_gap_days=1` and that row is NaN (edge, but consistent) |
| `min_gap_days <= 0` | raises `KuantValueError` via `require_positive` |
| `prices` not a DataFrame | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| Empty column set | returns `{}` |

## Examples

### Basic detection

```python
>>> import pandas as pd
>>> import numpy as np
>>> from kuant.backtest.lifecycle import detect_delistings
>>> idx = pd.date_range("2020-01-01", periods=20, freq="D")
>>> panel = pd.DataFrame({
...     "STILL_LIVE":     np.arange(20, dtype=float),
...     "DELISTED":       list(range(10)) + [np.nan] * 10,
...     "HALTED_ONE_DAY": [1.0] * 9 + [np.nan] + [1.0] * 10,
... }, index=idx)
>>> detected = detect_delistings(panel, min_gap_days=5)
>>> sorted(detected)
['DELISTED']
>>> str(detected["DELISTED"])
'2020-01-10'
```

`STILL_LIVE` never goes NaN. `HALTED_ONE_DAY` has an interior
one-row gap but resolves; its trailing-NaN count is zero. Only
`DELISTED` matches all three rules.

### Composed into the panel report

```python
>>> from kuant.backtest.lifecycle import (
...     lifecycles_from_panel,
...     lifecycle_panel_report,
...     TerminalAction,
... )
>>> lifecycles = lifecycles_from_panel(
...     panel,
...     min_gap_days=5,
...     terminal_action=TerminalAction.MARK_TO_ZERO,
... )
>>> sorted(lifecycles)
['DELISTED']
>>> report = lifecycle_panel_report(panel, lifecycles)
>>> float(report.terminal_returns["DELISTED"].sum())
-1.0
```

The heuristic detects one dead name; the report injects a `-1.0`
return on the row after the last real print (2020-01-11 here).

### Tuning `min_gap_days`

```python
>>> # A three-row trailing gap: caught with 3, missed with 5.
>>> idx = pd.date_range("2020-01-01", periods=10, freq="D")
>>> panel = pd.DataFrame({
...     "SHORT_TAIL": [1.0] * 7 + [np.nan] * 3,
... }, index=idx)
>>> detect_delistings(panel, min_gap_days=3)
{'SHORT_TAIL': datetime.date(2020, 1, 7)}
>>> detect_delistings(panel, min_gap_days=5)
{}
```

Business-day calendars typically want 3; raw calendar-day panels
typically want 5 or higher. Pick the value once for the panel and
stick with it; changing the threshold mid-analysis shifts the set
of flagged names.

### Composing with a hand-labeled mapping

```python
>>> from datetime import date
>>> from kuant.backtest.lifecycle import SecurityLifecycle, TerminalAction
>>> auto = lifecycles_from_panel(panel, min_gap_days=3)
>>> # Override the auto-detected entry with a known recovery ratio.
>>> auto["SHORT_TAIL"] = SecurityLifecycle(
...     symbol="SHORT_TAIL",
...     delisting_date=date(2020, 1, 7),
...     terminal_action=TerminalAction.PRORATE_RECOVERY,
...     terminal_recovery=0.45,
... )
>>> auto["SHORT_TAIL"].terminal_action.value
'prorate_recovery'
```

The auto mapping is a normal `dict` and merges cleanly with
per-symbol overrides.

## Cross-check tests

- `detect_delistings` on an all-live panel returns `{}`.
- `detect_delistings` on an all-NaN column returns `{}`; no last
  real print to anchor.
- `lifecycles_from_panel` composed with `apply_lifecycle_panel`
  masks the detected columns and leaves live columns untouched.
- Round-trip: a hand-built lifecycle for a delisting matches the
  output of `lifecycles_from_panel` on the same panel when the
  terminal action is identical.
- `min_gap_days` monotonicity: for the same panel,
  `detect_delistings(panel, k1)` is a superset of
  `detect_delistings(panel, k2)` whenever `k1 <= k2`.

## Related kernels

- `kuant.backtest.lifecycle.SecurityLifecycle`, `apply_lifecycle_panel`,
  `lifecycle_panel_report` ([`security.md`](security.md)): consume
  the mapping returned here.
- `kuant.data.align`: align feeds onto a common calendar before
  running the heuristic. A mis-aligned panel with per-column date
  offsets can produce spurious tails.
- `kuant.edgecases`: NaN policies for interior gaps that the
  heuristic deliberately does not flag as delistings.

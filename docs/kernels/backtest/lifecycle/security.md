# security - SecurityLifecycle and companion kernels

## Purpose

Encode the tradeable window and terminal-day fate of every security
in a panel as a typed record, and apply that record to prices,
returns, and simulator gates. Closes the silent-corruption gap where
naive engines mishandle NaN prices on delisted names; see
[README](README.md) for the framing.

## Public API

```python
from datetime import date
from kuant.backtest.lifecycle import (
    TerminalAction,
    SecurityLifecycle,
    LifecyclePanelResult,
    apply_lifecycle,
    apply_lifecycle_panel,
    lifecycle_returns,
    tradeable_mask,
    lifecycle_panel_report,
)
```

### `TerminalAction`

String-valued enum. Values round-trip through JSON and parquet
without a custom encoder.

| Value | Terminal day plus one return |
| --- | --- |
| `LIQUIDATE_AT_LAST` | `0.0` |
| `MARK_TO_ZERO` | `-1.0` |
| `PRORATE_RECOVERY` | `terminal_recovery - 1.0` |

### `SecurityLifecycle`

Frozen dataclass. One record per security.

```python
SecurityLifecycle(
    symbol: str,
    listing_date: date | None = None,
    delisting_date: date | None = None,
    terminal_action: TerminalAction = TerminalAction.MARK_TO_ZERO,
    terminal_recovery: float = 0.0,
)
```

- `listing_date=None` means "listed before the panel begins"; no
  pre-listing masking is applied.
- `delisting_date=None` means "still trading."
- `terminal_recovery` is only consulted when
  `terminal_action == PRORATE_RECOVERY`. Must lie in `[0, 1]`; the
  constructor validates via `require_probability`.

`.summary()` returns a short human-readable string.

### Kernels

- `tradeable_mask(index, lifecycle) -> np.ndarray[bool]`
- `apply_lifecycle(prices: pd.Series, lifecycle) -> pd.Series`
- `apply_lifecycle_panel(prices: pd.DataFrame, lifecycles: Mapping[str, SecurityLifecycle]) -> pd.DataFrame`
- `lifecycle_returns(prices: pd.Series, lifecycle) -> pd.Series`
- `lifecycle_panel_report(prices: pd.DataFrame, lifecycles) -> LifecyclePanelResult`

### `LifecyclePanelResult`

Dataclass returned by `lifecycle_panel_report`. Fields:

- `cleaned`: `pd.DataFrame` of prices with pre-listing and
  post-delisting rows nulled per column.
- `tradeable`: boolean `pd.DataFrame`; True where an order could
  have filled.
- `terminal_returns`: `pd.DataFrame` of the same shape, all zeros
  except at each column's terminal day plus one row, which carries
  the action-specific return.
- `lifecycles`: copy of the input mapping.
- `.summary()`: short human-readable string.
- `.to_parquet(path)`: write `cleaned` to parquet (requires pyarrow).
  Only prices serialize; reconstruct the mask and terminal-return
  frames by re-running the kernels against the original mapping.

## Design decisions

### 1. Tradeable window, not NaN, gates orders

Simulators should call `tradeable_mask` and use the boolean panel to
decide whether an order fills. NaN prices are a downstream
consequence of masking, not the primary signal. Two reasons:

1. NaN can appear for reasons that are not delistings (mid-day
   halts, exchange quote gaps, corrupted vendor rows). Conflating
   these with delisting silently mis-classifies live names.
2. NaN cannot express the difference between "not yet listed",
   "temporarily missing", and "gone forever." The lifecycle
   distinguishes the three.

Concretely, a naive simulator that gates on `~np.isnan(price)`
cannot tell a listing gap from a halt, and cannot inject the
terminal return at all.

### 2. Terminal day plus one, not terminal day, carries the fate

`LIQUIDATE_AT_LAST`, `MARK_TO_ZERO`, and `PRORATE_RECOVERY` all
inject their return on the first index row strictly after
`delisting_date`. Rationale: the reported close on the delisting
date is a real trade; the return computed from it (against the
previous close) is a real return. The "what happens to the
position" event is the transition from that close to whatever
comes next (zero, cash, or a partial recovery).

Convention: `LIQUIDATE_AT_LAST` writes `0.0` on that row, not NaN.
The semantic is "position closed at close, cash held, no move."
Callers who prefer to drop the row entirely can filter on
`tradeable_mask` AFTER consuming the terminal return.

### 3. `PRORATE_RECOVERY` uses `r - 1`, not `r`

`terminal_recovery` is a fraction of previous close recovered.
A cash-plus-stock deal that returns 40 cents on the dollar has
`terminal_recovery=0.40`, and the return on the transition row is
`0.40 - 1.0 = -0.60`. This matches how typical vendor
delisting-return code schemas encode partial recoveries: the
recovery divided by the prior close, minus one.

### 4. `apply_lifecycle` masks only; `lifecycle_returns` injects

Deliberate split. `apply_lifecycle` returns cleaned prices, and
downstream kuant kernels (rolling volatility, moving averages,
correlation) consume the cleaned prices. Those kernels must NOT
see a synthetic terminal day; a `-1.0` return would dominate any
rolling std window.

`lifecycle_returns` composes the mask with the terminal injection
in a single call and is the right thing to feed into a PnL
aggregator.

### 5. `apply_lifecycle_panel` never adds columns

If `lifecycles` names a symbol not in `prices.columns`, the entry is
silently ignored. If `prices` has a column with no lifecycle entry,
the column is passed through untouched. The kernel refuses to
synthesize empty columns because doing so would mask upstream
data-loading bugs; the caller expected a name in the panel and it
is missing.

### 6. Frozen dataclass, string-valued enum

`SecurityLifecycle` is `frozen=True`. A lifecycle is a fact about
history, not a mutable buffer. `TerminalAction` inherits `str`, so
`json.dumps({"action": TerminalAction.MARK_TO_ZERO.value})` and
parquet round-trips work without a custom serializer.

### 7. Constructor validates ordering and recovery

`listing_date > delisting_date` raises `KuantValueError` with a
`KE-VAL-RANGE` code. `terminal_recovery` outside `[0, 1]` raises
via `require_probability`. Fail early rather than produce a
malformed return series later.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `listing_date=None, delisting_date=None` | tradeable everywhere; kernels are effectively no-ops |
| `delisting_date` after last index row | no terminal return injected; last real print kept |
| `delisting_date` before first index row | entire series masked to NaN |
| `listing_date > delisting_date` | raises `KuantValueError [KE-VAL-RANGE]` |
| `terminal_recovery` outside `[0, 1]` | raises `KuantValueError` via `require_probability` |
| `apply_lifecycle` on non-Series | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| `apply_lifecycle_panel` mapping names an absent column | column skipped silently |
| Panel column has no mapping entry | column passed through untouched |
| No post-delisting index row exists | no terminal return injected; last real return preserved |

## Examples

### Single series, mark to zero

```python
>>> import pandas as pd
>>> from datetime import date
>>> from kuant.backtest.lifecycle import (
...     SecurityLifecycle, TerminalAction,
...     apply_lifecycle, lifecycle_returns, tradeable_mask,
... )
>>> idx = pd.date_range("2020-01-01", periods=6, freq="D")
>>> prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], index=idx)
>>> lc = SecurityLifecycle(
...     symbol="DEAD",
...     delisting_date=date(2020, 1, 3),
...     terminal_action=TerminalAction.MARK_TO_ZERO,
... )
>>> apply_lifecycle(prices, lc).tolist()
[10.0, 11.0, 12.0, nan, nan, nan]
>>> lifecycle_returns(prices, lc).round(4).tolist()
[nan, 0.1, 0.0909, -1.0, nan, nan]
>>> tradeable_mask(idx, lc).tolist()
[True, True, True, False, False, False]
```

The last live close is on 2020-01-03. The terminal day plus one
return of `-1.0` lands on 2020-01-04. All later rows are NaN.

### Prorated recovery

A cash-plus-stock deal returns 30 cents on the dollar:

```python
>>> lc = SecurityLifecycle(
...     symbol="MERGED",
...     delisting_date=date(2020, 1, 3),
...     terminal_action=TerminalAction.PRORATE_RECOVERY,
...     terminal_recovery=0.30,
... )
>>> lifecycle_returns(prices, lc).round(4).tolist()[3]
-0.7
```

The `-0.7` is `0.30 - 1.0`. The pre-terminal in-window returns are
identical to the `MARK_TO_ZERO` case.

### Liquidate at last

```python
>>> lc = SecurityLifecycle(
...     symbol="TAKEN_PRIVATE",
...     delisting_date=date(2020, 1, 3),
...     terminal_action=TerminalAction.LIQUIDATE_AT_LAST,
... )
>>> lifecycle_returns(prices, lc).round(4).tolist()[3]
0.0
```

Position closed at close; no move on the transition row.

### End-to-end panel report

Two symbols; one live, one delisted mid-panel.

```python
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import date
>>> from kuant.backtest.lifecycle import (
...     SecurityLifecycle, TerminalAction,
...     lifecycle_panel_report,
... )
>>> idx = pd.date_range("2020-01-01", periods=6, freq="D")
>>> panel = pd.DataFrame({
...     "LIVE":     [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
...     "DELISTED": [20.0, 21.0, 22.0, np.nan, np.nan, np.nan],
... }, index=idx)
>>> lifecycles = {
...     "LIVE": SecurityLifecycle(symbol="LIVE"),
...     "DELISTED": SecurityLifecycle(
...         symbol="DELISTED",
...         delisting_date=date(2020, 1, 3),
...         terminal_action=TerminalAction.MARK_TO_ZERO,
...     ),
... }
>>> report = lifecycle_panel_report(panel, lifecycles)
>>> report.cleaned["DELISTED"].tolist()
[20.0, 21.0, 22.0, nan, nan, nan]
>>> report.tradeable["DELISTED"].tolist()
[True, True, True, False, False, False]
>>> report.terminal_returns["DELISTED"].tolist()
[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
>>> bool(report.tradeable["LIVE"].all())
True
>>> print(report.summary())
=== LifecyclePanelResult ===
panel shape:        (6, 2)
n symbols:          2
n with delisting:   1
```

The live column is untouched. The delisted column has its
post-2020-01-03 rows masked to NaN, the tradeable mask flips to
False on the same rows, and the terminal-return frame carries the
`-1.0` on the row after the last live close (2020-01-04).

## Cross-check tests

- Round-trip: `apply_lifecycle` followed by `pct_change` on the
  in-window slice matches `lifecycle_returns` up to the terminal
  row.
- All three `TerminalAction` variants produce the expected value on
  the terminal-day-plus-one row; rows strictly after that row are
  NaN.
- Frozen dataclass equality: two `SecurityLifecycle` records with
  the same fields compare equal and hash identically.
- `apply_lifecycle_panel` with an empty mapping is a no-op modulo
  the float64 upcast; column dtypes downstream are consistent.
- `lifecycle_panel_report(panel, {}).cleaned` equals
  `panel.astype(float)`.

## Related kernels

- `kuant.backtest.lifecycle.detect_delistings` and `lifecycles_from_panel`
  ([`detect.md`](detect.md)): heuristic inference when a real
  delisting table is not at hand.
- `kuant.data.align`: align raw feeds onto a common calendar before
  applying lifecycles.
- `kuant.stats.rollstd`, `kuant.stats.rollmean`: consume the
  `cleaned` frame from `LifecyclePanelResult` for rolling aggregates
  that must not see the synthetic terminal row.

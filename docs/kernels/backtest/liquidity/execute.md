# execute - execute_fill, execute_fill_panel, liquidity_mask

## Purpose

Translate a requested `(size, price)` order into an actual fill,
respecting participation cap, minimum lot, and size-dependent
slippage. Return the outcome as a typed `FillResult` whose
categorical `reason` distinguishes a full fill, a partial fill, and
each of the rejection modes. Also expose a boolean `liquidity_mask`
kernel that gates untradeable dates for composition with
`kuant.backtest.lifecycle.tradeable_mask`.

## Public API

```python
from kuant.backtest.liquidity import (
    FillResult,
    execute_fill,
    execute_fill_panel,
    liquidity_mask,
)
```

### `FillResult`

Dataclass returned by `execute_fill`.

```python
FillResult(
    price: float,
    size_filled: float,
    size_rejected: float,
    slippage_bps: float,
    reason: str,
    cost: float,
)
```

Attributes:

- `price`: actual fill price after slippage. `NaN` if the order was
  rejected.
- `size_filled`: signed quantity filled (`+` buy, `-` sell). Zero if
  fully rejected.
- `size_rejected`: signed quantity NOT filled. Simulators may
  re-queue this onto the next bar or drop it, depending on strategy.
- `slippage_bps`: applied slippage in basis points. Zero if fully
  rejected.
- `reason`: categorical outcome; see the table below.
- `cost`: total transaction value, `abs(size_filled) * price`. Zero
  if rejected.

`.summary()` returns a short human-readable string.

### `execute_fill`

```python
execute_fill(
    size: float,
    price: float,
    profile: LiquidityProfile,
    timestamp,
    model,
) -> FillResult
```

Single-order fill attempt. `size` is signed (`+` buy, `-` sell),
`price` is the pre-slippage reference (mid or close), `timestamp` is
a date-like value looked up in `profile.adv_series`, and `model` is
any object exposing `compute_slippage(size, adv, side) -> float`.

### `execute_fill_panel`

```python
execute_fill_panel(
    orders: pd.DataFrame,
    profile: LiquidityProfile,
    model,
) -> pd.DataFrame
```

Batch loop over `execute_fill` for one profile. `orders` must have
columns `timestamp`, `size`, `price`. Returns a DataFrame with the
requested columns plus `fill_price`, `size_filled`, `size_rejected`,
`slippage_bps`, `reason`, `cost`.

### `liquidity_mask`

```python
liquidity_mask(
    index,
    profile: LiquidityProfile,
    min_adv: float = 0.0,
) -> np.ndarray[bool]
```

Boolean mask over `index`: True where `profile.adv_series[d]` is
finite and strictly greater than `min_adv`. Defaults to `0.0`, which
gates out only NaN and non-positive ADV.

## Reason codes

| `reason` | Meaning | `size_filled` | `size_rejected` | `price` |
| --- | --- | --- | --- | --- |
| `OK` | Fully filled, size fit under the ADV cap and cleared min_size | `size` | `0` | filled |
| `CAPPED_PARTICIPATION` | Partial fill; requested size exceeded `max_participation * ADV_t`, truncated to the cap | `side * cap` | `size - size_filled` | filled |
| `BELOW_MIN_SIZE` | Fully rejected; requested absolute size below `profile.min_size` | `0` | `size` | `NaN` |
| `NO_LIQUIDITY` | Fully rejected; ADV on `timestamp` is NaN or non-positive | `0` | `size` | `NaN` |
| `MISSING_DATE` | Fully rejected; `timestamp` not found in `profile.adv_series.index` | `0` | `size` | `NaN` |

The reason is intentionally a plain string, not an enum: the same
value is compared inside the fill layer, serialised to the returned
DataFrame, and translated into an `OrderStatus` by
`kuant.backtest.fill.submit_order`. A string keeps the wire format
stable across those boundaries without a shared enum import.

## Design decisions

### 1. Truncate to cap, not reject

An order larger than `max_participation * ADV_t` is truncated to the
cap and reported as `CAPPED_PARTICIPATION`, with the unfilled
quantity in `size_rejected`. The alternative, fully rejecting the
order, would force every strategy that ever prints an oversize
signal to duplicate the participation-cap arithmetic upstream. A
truncation keeps the fill layer authoritative on capacity while
letting the caller decide whether to re-queue the residual, drop it,
or slice it across bars.

The residual is signed: a rejected 400 lot on a 250 lot cap for a
buy reports `size_filled=+250`, `size_rejected=+150`. Sums across a
day of fills reconcile trivially.

### 2. Reject below min_size, do not silently round

An order whose absolute size is below `profile.min_size` returns
`BELOW_MIN_SIZE` with zero filled. Silently rounding up to `min_size`
would over-execute a strategy's intent; silently rounding down to
zero would under-execute it and mask the constraint. Reporting the
rejection lets the caller decide whether to aggregate several small
orders into one bar or hold the intent as unfilled.

Fractional-share brokers relax this constraint in the real world;
model it by setting `min_size` to a fractional value at construction
rather than by circumventing the check.

### 3. `NO_LIQUIDITY` and `MISSING_DATE` are distinct rejections

A NaN or zero ADV on a date that IS in the index (a halt, a
zero-volume bar) reports `NO_LIQUIDITY`. A date that is NOT in the
index at all reports `MISSING_DATE`. The two look identical to a
caller who only reads `reason == "OK"`, but they diagnose different
upstream bugs:

- `NO_LIQUIDITY` at scale means the ADV feed carries genuine zero
  bars the strategy is trying to trade through; the fix is upstream
  liquidity gating with `liquidity_mask`.
- `MISSING_DATE` at scale means the strategy and the profile are on
  different calendars; the fix is data alignment.

### 4. Positive-finite price and finite size, checked at entry

`execute_fill` raises `KuantValueError` on non-finite or
non-positive `price` and on non-finite `size`. Zero and NaN prices
are the caller's responsibility to gate out (via `tradeable_mask`);
the fill kernel refuses to guess whether they mean "delisted",
"halted", or "corrupt vendor row." NaN size is a strategy-layer bug
and should not silently produce a zero fill.

### 5. `liquidity_mask` uses strict `>` on `min_adv`

The mask sets True where `adv > min_adv`, not `>=`. Consequence:
`min_adv=0.0` gates out exactly the bars where ADV is zero or
negative; it does not admit them. A caller who wants "at least X
shares" passes `min_adv=X-1` or accepts the strict inequality.

The kernel emits `KW-LIQ-MASK-ALL-FALSE` when the input index is
non-empty and every row masks out. That case usually means the ADV
feed was fed in the wrong units (dollars instead of shares) or the
profile's index does not intersect the query index; both are silent
bugs otherwise.

### 6. `execute_fill_panel` is a per-row loop in v1

The panel kernel is a plain iterating loop over `execute_fill`. The
API is stable; a future version will vectorize the ADV lookup and
slippage computation for speed. Unrecognised columns on the input
frame trigger `KW-FILL-PANEL-EXTRA-COLS` and are silently dropped
from the output; the caller can stitch them back on afterwards.

An empty `orders` frame emits `KW-FILL-PANEL-EMPTY` and returns an
empty result, since it usually indicates the upstream strategy
failed to emit any orders on the batch rather than a deliberate
no-op.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `price <= 0` or non-finite | raises `KuantValueError [KE-VAL-POSITIVE]` |
| `size` non-finite | raises `KuantValueError [KE-FILL-SIZE-NAN]` |
| `model` missing `compute_slippage` | raises `KuantValueError [KE-VAL-CONTRACT]` |
| `timestamp` not in `profile.adv_series.index` | `FillResult(reason="MISSING_DATE")` |
| ADV is NaN or non-positive at `timestamp` | `FillResult(reason="NO_LIQUIDITY")` |
| `abs(size) < profile.min_size` | `FillResult(reason="BELOW_MIN_SIZE")` |
| `abs(size) > profile.max_participation * ADV_t` | `FillResult(reason="CAPPED_PARTICIPATION")`, truncated |
| `orders` is not a `pd.DataFrame` | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| `orders` missing required columns | raises `KuantValueError [KE-VAL-SCHEMA]` |
| `orders` has extra columns | warns `KW-FILL-PANEL-EXTRA-COLS`, silently dropped |
| `orders` is empty | warns `KW-FILL-PANEL-EMPTY`, returns empty frame |
| `min_adv < 0` on `liquidity_mask` | raises `KuantValueError [KE-LIQ-MASK-MIN-ADV-NEGATIVE]` |
| `liquidity_mask` all-False on non-empty input | warns `KW-LIQ-MASK-ALL-FALSE` |

## Examples

### Fully filled order

```python
>>> import pandas as pd
>>> from datetime import date
>>> from kuant.backtest.liquidity import (
...     LiquidityProfile, FlatSlippage, execute_fill,
... )
>>> adv = pd.Series([1_000_000.0], index=[pd.Timestamp("2020-01-02")])
>>> profile = LiquidityProfile(
...     symbol="XYZ", adv_series=adv, min_size=100.0,
...     max_participation=0.10,
... )
>>> r = execute_fill(1000.0, 50.0, profile, date(2020, 1, 2),
...                  FlatSlippage(bps=5))
>>> r.reason
'OK'
>>> round(r.price, 4)
50.025
>>> r.size_filled
1000.0
>>> r.size_rejected
0.0
```

A 1000-share buy at 50.0 with 5 bps flat slippage fills at
`50.0 * 1.0005 = 50.025`. The order is 0.1% of ADV, well under the
10% cap.

### Partial fill capped by participation

```python
>>> r = execute_fill(250_000.0, 50.0, profile, date(2020, 1, 2),
...                  FlatSlippage(bps=5))
>>> r.reason
'CAPPED_PARTICIPATION'
>>> r.size_filled
100000.0
>>> r.size_rejected
150000.0
>>> round(r.price, 4)
50.025
```

The requested 250,000 shares would consume 25% of ADV, above the
10% cap. `execute_fill` truncates the filled quantity to
`0.10 * 1_000_000 = 100_000` and reports the 150,000 residual on
`size_rejected`. The slippage rate applies to the FILLED size, not
the requested size.

### Rejected below min_size

```python
>>> r = execute_fill(50.0, 50.0, profile, date(2020, 1, 2),
...                  FlatSlippage(bps=5))
>>> r.reason
'BELOW_MIN_SIZE'
>>> r.size_filled
0.0
>>> r.size_rejected
50.0
>>> import math; math.isnan(r.price)
True
>>> r.cost
0.0
```

The 50-share order is below the 100-share minimum; the fill is
refused outright, price is `NaN`, and the full requested size is
carried on `size_rejected`.

### Missing date vs no liquidity

```python
>>> r = execute_fill(1000.0, 50.0, profile, date(2020, 1, 3),
...                  FlatSlippage(bps=5))
>>> r.reason
'MISSING_DATE'
```

`2020-01-03` is not in the single-row `adv_series` above; the fill
is refused with `MISSING_DATE`. Compare with a date that is
present but carries `NaN`:

```python
>>> import numpy as np
>>> adv2 = pd.Series(
...     [1_000_000.0, np.nan],
...     index=pd.date_range("2020-01-02", periods=2, freq="D"),
... )
>>> profile2 = LiquidityProfile(
...     symbol="XYZ", adv_series=adv2, min_size=100.0,
...     max_participation=0.10,
... )
>>> execute_fill(1000.0, 50.0, profile2, date(2020, 1, 3),
...              FlatSlippage(bps=5)).reason
'NO_LIQUIDITY'
```

### liquidity_mask

```python
>>> import pandas as pd
>>> from kuant.backtest.liquidity import LiquidityProfile, liquidity_mask
>>> idx = pd.date_range("2020-01-01", periods=3, freq="D")
>>> adv = pd.Series([1_000_000.0, 900_000.0, 1_200_000.0], index=idx)
>>> profile = LiquidityProfile(symbol="XYZ", adv_series=adv)
>>> liquidity_mask(idx, profile, min_adv=0.0).tolist()
[True, True, True]
>>> liquidity_mask(idx, profile, min_adv=1_000_000.0).tolist()
[False, False, True]
```

The strict-`>` comparison at the 1,000,000 floor rejects the first
row (equal to the floor) as well as the second (below).

### Compose with lifecycle

```python
>>> from kuant.backtest.lifecycle import (
...     SecurityLifecycle, tradeable_mask,
... )
>>> lc = SecurityLifecycle(symbol="XYZ")
>>> can_trade = tradeable_mask(idx, lc) & liquidity_mask(idx, profile)
>>> can_trade.tolist()
[True, True, True]
```

The tradeable window AND the liquidity floor define "can this order
fill today?"

## Cross-check tests

- Round-trip: for a fully-filled buy, `price * (1 + slippage_bps /
  10_000)` equals the returned `fill_price` up to floating-point
  rounding.
- Partial-fill accounting: `size_filled + size_rejected == size` for
  every reason value; sign is preserved.
- Rejection returns: `price` is `NaN` and `cost` is `0.0` for every
  reason other than `OK` and `CAPPED_PARTICIPATION`.
- `execute_fill_panel` output row order equals input row order.
- `liquidity_mask` on an index that does not intersect the profile
  returns all-False and emits `KW-LIQ-MASK-ALL-FALSE`.

## Related kernels

- [`profile.md`](profile.md): `LiquidityProfile` supplies the
  microstructure inputs.
- [`models.md`](models.md): `FlatSlippage`, `LinearImpact`,
  `SquareRootImpact` fill the `model` argument.
- `kuant.backtest.lifecycle.tradeable_mask`: parallel gate on the
  listing window; compose with `&`.
- `kuant.backtest.fill.submit_order`: consumes `FillResult` and
  maps `reason` into an `OrderStatus` for order reconciliation.

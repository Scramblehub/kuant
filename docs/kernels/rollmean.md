# rollmean — Rolling window mean

## Purpose

`rollmean(x, w)[i] = mean(x[i-w+1 : i+1])` for `i ≥ w-1`; NaN for `i < w-1`.

Foundation for every rolling statistic in kuant. Used by `rollstd`,
`zscore`, and downstream signal work.

## Public API

```python
from kuant.stats import rollmean

result = rollmean(x, window)
```

- `x` — 1D numpy or cupy array. Ints promote to float64.
- `window` — positive int. Must be ≤ len(x) for non-trivial output.
- Returns 1D array, same length/backend/dtype.

## Design decisions

### 1. Cumulative-sum trick — O(n), independent of window

```math
csum[i]        = sum(x[0..i])
window_sum[i]  = csum[i] - csum[i-w]
rollmean[i]    = window_sum[i] / w
```

Naive rolling is O(n·w). The cumsum trick reduces to O(n) — one forward
pass, then differences. Window size stops mattering for wall-clock.

### 2. Strict-window NaN policy

If ANY value in the window is NaN, output NaN for that window. Matches
pandas `rolling(w, min_periods=w).mean()`.

Implementation runs TWO parallel cumsums:
- `cumsum(x_safe)` where NaNs are replaced with 0
- `cumsum(is_nan.astype(int))` — the running count of NaNs

The rolling NaN count tells us which windows to invalidate. A single
`where(count == 0, mean, nan)` finishes it.

This costs one extra cumsum and one extra concatenate — cheap versus the
naive alternative of guarding each window individually.

### 3. First `w-1` entries are NaN

Convention: no partial windows. Matches pandas `min_periods=w`. Simplifies
downstream code — every "valid" entry has exactly `w` inputs behind it.

### 4. Preserves backend and dtype

Same contract as `kuant.core`: cupy in → cupy out, float32 in → float32
out, int in → float64 out.

### 5. Prepend zero to `cumsum` for clean indexing

Rather than special-casing `i-w < 0`, we prepend a 0 to the cumsum so that
`window_sum = csum[w:] - csum[:-w]` works for all valid indices. One
`concatenate` is cheaper than a branch in the hot loop.

## Edge cases

| Condition | Output |
|---|---|
| `window == 1` | identity (each value is its own window mean) |
| `window == len(x)` | first `n-1` NaN, last is overall mean |
| `window > len(x)` | all NaN |
| `window ≤ 0` | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| Empty input | empty output |
| Single NaN at index `k` | poisons `w` windows overlapping `k`, then math resumes |
| All-NaN input | all NaN output |

## Cross-check tests

- `test_matches_pandas_uniform` — 1000-point match to pandas rolling mean
- `test_matches_pandas_with_nans` — same, with 5% scattered NaNs
- `test_matches_naive_loop` — cumsum trick matches O(n·w) direct computation

The pandas cross-check is the strongest: any drift in our NaN policy or
indexing shows up immediately against the industry-standard reference.

## Test coverage (23 tests)

Golden (4 hand-computed), pandas reference (uniform + with NaNs), edge
cases (window=1/=n/>n/≤0, 2D input, isolated NaN, all NaN, dtype, list
input), property tests (first w-1 NaN, length preserved, matches naive
loop), CPU==GPU parity (clean + with NaNs), backend preservation.

## Direct usage in kuant

Backbone for every rolling statistic. Direct use case in the M9 monitor:
compute `rollmean(volume, 20)` as the "typical volume" baseline, then
detect anomalies as deviations.

## Related kernels

- `kuant.stats.rollstd` — sibling; will share the NaN-handling pattern
- `kuant.stats.zscore` — composes rollmean and rollstd

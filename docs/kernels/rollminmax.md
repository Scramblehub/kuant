# rollmin / rollmax — Rolling window extremes

## Purpose

`rollmin(x, w)[i]` = min of `x[i-w+1 : i+1]`
`rollmax(x, w)[i]` = max of `x[i-w+1 : i+1]`

Foundation for:

- Rolling channel indicators (Donchian, high-low bands)
- Range-based stops / take-profits (trailing highs)
- Regime detection (rolling drawdown = current - rolling max)

## Public API

```python
from kuant.stats import rollmin, rollmax

lows = rollmin(x, window)
highs = rollmax(x, window)
```

Same signature contract as `rollmedian`.

## Design decisions

### 1. Sliding-view + `xp.min` / `xp.max` — same pattern as rollquantile

```python
windowed = sliding_window_view(x, w)     # (n-w+1, w)
result   = xp.min(windowed, axis=1)      # or xp.max
```

Zero-copy strided view on numpy; may materialize on cupy.

### 2. Shared reducer helper

`_reduce_over_windows(x, w, 'min' | 'max')` factors out the sliding-view

- reshape logic. Both `rollmin` and `rollmax` are one-line delegations
to it. Trivial to extend to `argmin`/`argmax` later if needed.

### 3. Strict NaN policy for free

`np.min` / `np.max` propagate NaN by default. Any NaN in the window
gives NaN output. No extra masking needed.

### 4. Preserves backend / dtype / int-promotion

Same contract as the rest of kuant.stats.

## Edge cases

| Condition | Output |
| --- | --- |
| `window == 1` | identity |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| NaN in window | NaN |
| Monotonically increasing input | `rollmax[i] == x[i]`, `rollmin[i] == x[i-w+1]` |

## Cross-check tests

- `test_rollmin_matches_pandas` / `test_rollmax_matches_pandas` — 500 random points
- `test_matches_pandas_with_nans` — NaN behavior matches pandas
- `test_min_neg_max_neg` — `min(-x) == -max(x)`
- `test_min_leq_max` — `rollmin <= rollmax` at every valid index
- `test_monotonically_increasing_input` — analytic sanity check

## Test coverage (18 tests)

Golden (hand-computed min/max of a small example, window=1 identity),
pandas reference (uniform + NaNs), edge cases (window bounds, 2D input,
NaN, dtype), property tests (min ≤ max, negation symmetry, first w-1
NaN, monotonic-input analytic check), CPU==GPU parity.

## Direct usage in kuant

- Rolling 252-day high for trailing-stop logic (M9 monitor)
- Rolling 60-day low for oversold detection
- `x - rollmax(x, w)` as rolling drawdown from peak
- `(x - rollmin(x, w)) / (rollmax(x, w) - rollmin(x, w))` as
  normalized position in range (Stochastic %K style)

## Related kernels

- `kuant.stats.rollquantile` — sibling sliding-view kernel; `q=0` and
  `q=1` give the same values as `rollmin` and `rollmax` (but computed
  via sort rather than linear scan)
- **Future**: `kuant.stats.rollargmin`, `kuant.stats.rollargmax` — index
  of the extreme within each window; would use `xp.argmin` / `xp.argmax`
  and add `w - 1` corrections for absolute-index output

# rollquantile — Rolling window quantile

## Purpose

`rollquantile(x, w, q)[i]` = q-quantile of `x[i-w+1 : i+1]` for
`q ∈ [0, 1]`.

Also exports:

- `rollmedian(x, w)` = `rollquantile(x, w, 0.5)`
- `rollpercentile(x, w, p)` = `rollquantile(x, w, p/100)` for `p ∈ [0, 100]`

Underlies rank-based signals, robust dispersion measures (IQR),
threshold indicators, and any statistic that resists tail contamination
better than mean/std.

## Public API

```python
from kuant.stats import rollquantile, rollmedian, rollpercentile

result = rollquantile(x, window, q)         # q in [0, 1]
result = rollmedian(x, window)              # equivalent to q=0.5
result = rollpercentile(x, window, p)       # p in [0, 100]
```

## Design decisions

### 1. Sliding-window view + `xp.quantile(axis=1)`

Quantiles don't decompose additively — no cumsum trick works.
Instead:

```python
windowed = sliding_window_view(x, w)      # shape (n-w+1, w)
q_per_win = xp.quantile(windowed, q, axis=1)
```

On numpy, `sliding_window_view` returns a strided view (zero copy).
On cupy, the equivalent function may materialize the 2D array,
depending on cupy version.

**Cost model:**

- Memory: O(n·w) view (or materialized, on some cupy versions)
- Compute: O(n·w log w) — one sort per window
- For typical windows (≤ 500) on 100k-element inputs, this is fast

For very large `w × n` products (e.g. w=1000, n=1M), consider
chunking the input through the throttle. Not implemented in V1.

### 2. Strict NaN policy — free via numpy semantics

`np.quantile` propagates NaN (since 1.24). If any value in the
window is NaN, the quantile is NaN. Same on cupy. Matches
`rollmean` / `rollstd` semantics without extra code.

### 3. `rollmedian` and `rollpercentile` are thin wrappers

Same implementation, different call-site ergonomics. Median is a
common enough operation that having a named function is worth the
one-line wrapper.

### 4. Interpolation method: numpy default (linear)

`xp.quantile` uses linear interpolation between adjacent
order-statistics. This matches pandas' rolling().quantile()
default. Alternative methods (`nearest`, `lower`, `higher`) can
be added if a use case emerges.

## Edge cases

| Condition | Output |
| --- | --- |
| `window == 1` | identity (each value is its own single-element quantile) |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `q < 0` or `q > 1` | raises `ValueError` |
| `p < 0` or `p > 100` (percentile) | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| NaN in window | NaN (inherited from `np.quantile`) |

## Cross-check tests

- `test_median_matches_pandas` — 500 random points, `atol=1e-12`
- `test_quantile_matches_pandas` — 4 q values (10, 25, 75, 90) matched to pandas
- `test_quantile_matches_pandas_with_nans` — NaN behavior matches
- `test_percentile_50_matches_median` — internal consistency
- `test_quantile_0_is_min`, `test_quantile_1_is_max` — endpoint semantics
- `test_quantile_monotonic_in_q` — non-decreasing in `q`
- `test_shift_by_constant` — quantile is shift-equivariant

## Test coverage (24 tests)

Golden (median of progression, min/max via q=0/1, percentile==median),
pandas reference (4 quantile levels + NaN case), edge cases (window
bounds, q/p bounds, 2D input, NaN, dtype), property tests (monotonic
in q, shift equivariance, length), CPU==GPU parity.

## Direct usage in kuant

- Robust dispersion via IQR: `rollpercentile(x, w, 75) - rollpercentile(x, w, 25)`
- Rank-normalized signals: check whether current value is above the
  rolling 90th percentile
- V8 sleeve gating: rolling-median filter for outlier price prints

## Related kernels

- `kuant.stats.rollmean` — additive rolling stat via cumsum
- `kuant.stats.rollstd` — sibling dispersion measure (parametric)
- **Future**: `kuant.stats.rollrank` — rank of current value within window
  (uses similar sliding-view + argsort)

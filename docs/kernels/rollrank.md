# rollrank — Rolling rank of the current value within its window

## Purpose

`rollrank(x, w)[i]` = 1-based rank of `x[i]` within `x[i-w+1 : i+1]`.

Answers "is today extreme relative to its recent history?" without
assuming any distribution. Signals built on ranks are robust to
outliers and to changes in scale/level of the underlying series.

## Public API

```python
from kuant.stats import rollrank

result = rollrank(x, window, pct=False)
```

- `pct=False` (default): raw 1-based rank in `[1, w]` (or fractional for
  ties)
- `pct=True`: `rank / w` in `(0, 1]`. Matches pandas `rolling.rank(pct=True)`.

## Design decisions

### 1. Count-based algorithm — no per-window sort

For each window, we count how many values are strictly less than the
last value, plus how many are equal (including the last value itself),
then combine:

```math
rank = less + (equal + 1) / 2
```

- `less` alone gives the ordinal rank of the smallest tied value.
- Adding `(equal + 1) / 2` averages across the tied positions.
- For unique values (`equal = 1`), this collapses to `less + 1`.

Compared to sort-based rank (`argsort` + lookup), this is O(n·w) per
level rather than O(n·w log w), and vectorizes cleanly on both numpy
and cupy.

### 2. Average-rank tie handling — matches pandas

pandas' `.rank()` defaults to average-rank for ties. Same convention
here so `rollrank` is a drop-in replacement.

### 3. NaN handling — explicit row mask

NaN never compares true in either direction (`NaN < x` and `NaN == x`
both False), so a naive count would silently ignore NaN values in the
window and report a wrong rank. Explicit `xp.any(xp.isnan(windowed),
axis=1)` mask catches every NaN-containing row and sets the output to
NaN. Matches the rest of kuant.stats' strict-window semantics.

### 4. Sliding-window view — same pattern as rollminmax / rollquantile

Zero-copy on numpy; may materialize on cupy depending on version.

### 5. `pct=True` normalizes by `w`, not by `w-1`

Matches pandas: `rank / count_of_non_null`. For a full non-NaN window,
`count = w`, so `pct = rank / w ∈ (0, 1]`. Not `(rank - 1) / (w - 1)`.

## Edge cases

| Condition | Output |
| --- | --- |
| `window == 1` | all 1.0 (each value is its own rank-1 window) |
| `window == 1, pct=True` | all 1.0 (rank/w = 1/1 = 1) |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| NaN in window | NaN |
| All-equal window | `(w + 1) / 2` (average of ranks 1..w) |

## Cross-check tests

- `test_matches_pandas_uniform` — 500 continuous random values
- `test_matches_pandas_pct` — same with `pct=True`
- `test_matches_pandas_with_ties` — integer inputs force frequent ties
- `test_matches_pandas_with_nans` — NaN behavior matches
- `test_shift_invariance`, `test_scale_invariance_positive` — rank is
  invariant under strictly-monotonic transforms
- `test_negation_reverses_rank` — `rollrank(-x) + rollrank(x) = w + 1`

## Test coverage (24 tests)

Golden values (ascending/descending, ties, all-equal, pct
normalization), pandas reference (uniform + pct + ties + NaN), edge
cases (window bounds, 2D input, dtype), property tests (bounds,
shift/scale invariance, negation symmetry), CPU==GPU parity.

## Direct usage in kuant

- Rank-based momentum: "is today's return in the top decile of the
  last 60 days?" as a signal
- Non-parametric outlier gates: flag `rollrank > 0.99` or `< 0.01`
- Cross-sectional-style scoring against a rolling reference window

## Related kernels

- `kuant.stats.rollquantile` — inverse direction (value at fixed q)
- `kuant.stats.rollminmax` — extremes; `rollrank` at raw ranks 1 and `w`
- **Future**: `kuant.stats.rollargmin` / `rollargmax` — position of the
  extreme value within the window

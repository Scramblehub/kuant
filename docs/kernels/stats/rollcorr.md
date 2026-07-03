# rollcorr — Rolling Pearson correlation

## Purpose

`rollcorr(x, y, w)[i]` = Pearson correlation of `x[i-w+1 : i+1]` and
`y[i-w+1 : i+1]`.

Backbone for pairs signals, cross-asset lead-lag detection, regime
diagnostics, and any dispersion-normalized comparison of two rolling
windows.

## Public API

```python
from kuant.stats import rollcorr

result = rollcorr(x, y, window)
```

- `x`, `y` — 1D arrays of equal length.
- Result is 1D, same length/backend/dtype.
- No `ddof` — it cancels in the ratio.

## Design decisions

### 1. Cumsum trick with 5 running sums

```math
sum_x, sum_y, sum_xy, sum_x², sum_y²
```

each an O(n) cumsum. Window sums via `csum[w:] - csum[:-w]`.

Then per window:

```math
cov  = sum_xy - sum_x·sum_y/w
varx = sum_x² - sum_x²/w
vary = sum_y² - sum_y²/w
corr = cov / √(varx·vary)
```

The `1/(w-ddof)` normalization drops out of the ratio, so
`ddof` doesn't appear in the API.

### 2. Shifted cumsums for stability

Correlation is invariant under BOTH shift and positive scale.
Subtracting `x[0]` and `y[0]` keeps the cumulative sums small and
prevents catastrophic cancellation for price-scale inputs. Same
trick as `rollstd`.

### 3. Union NaN mask

If EITHER `x` or `y` has a NaN in the window, the correlation is
undefined for that window. A single running NaN count on
`isnan(x) | isnan(y)` handles both series simultaneously.

### 4. Zero-variance guard

If `rollstd_x == 0` or `rollstd_y == 0` (constant window in either
series), the correlation is undefined → NaN. Explicit guard on
`denom > 0` in the division.

### 5. Clip to `[-1, 1]`

Floating-point noise can push the result slightly outside the
theoretical bounds (e.g. `1.0000000000001`). `xp.clip` on the
output ensures downstream code sees a valid correlation coefficient.

### 6. `window < 2` returns all NaN

A single-element window has zero variance → correlation undefined.
We short-circuit for `window == 1` (and `window <= 0` raises).

## Edge cases

| Condition | Output |
| --- | --- |
| `window == 1` | all NaN (undefined) |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `x.ndim != 1` or `y.ndim != 1` | raises `ValueError` |
| `len(x) != len(y)` | raises `ValueError` |
| Zero variance in either series | NaN for that window |
| NaN in either series | NaN for windows overlapping the NaN |
| Perfectly correlated / anti-correlated | ±1 exactly (after clip) |

## Cross-check tests

- `test_matches_pandas_uniform` — 500 random pairs, `atol=1e-10`
- `test_matches_pandas_with_nans` — scattered NaNs in both series
- `test_matches_pandas_large_magnitude` — price-scale (~4000), `atol=1e-8`
- `test_symmetry` — `corr(x, y) == corr(y, x)`
- `test_shift_invariance` — `corr(x + a, y + b) == corr(x, y)`
- `test_scale_invariance_positive` — `corr(αx, βy) == corr(x, y)` for α, β > 0
- `test_scale_sign_flip` — `corr(x, -y) == -corr(x, y)`

## Test coverage (26 tests)

Golden (perfect ±1, orthogonal), pandas reference (uniform + NaNs +
price scale), edge cases (window bounds, length mismatch, 2D input,
NaN in either series, zero variance, dtype, list input), property
tests (symmetry, shift/scale invariance, sign flip, first w-1 NaN,
range [-1, 1]), CPU==GPU parity.

## Direct usage in kuant

- Cross-asset lead-lag diagnostics on returns streams
- Pairs signal for M9 basket-level correlation drift
- Rolling `corr(strategy_returns, benchmark_returns)` as a regime metric

## Related kernels

- `kuant.stats.rollmean`, `kuant.stats.rollstd` — share the cumsum-
  trick + shifted-stability pattern
- `kuant.stats.zscore` — same NaN-strict + zero-denom pattern
- **Future**: `kuant.stats.rollcov` (unnormalized covariance) if a use
  case emerges — currently absorbed inside rollcorr

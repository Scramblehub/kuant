# rollstd — Rolling window standard deviation

## Purpose

`rollstd(x, w, ddof=1)[i]` = std of `x[i-w+1 : i+1]` with `w - ddof`
degrees of freedom.

Foundation for `zscore`, Bollinger bands, rolling volatility, and any
dispersion-based signal.

## Public API

```python
from kuant.stats import rollstd

result = rollstd(x, window, ddof=1)
```

- `x` — 1D array. Ints promote to float64.
- `window` — positive int.
- `ddof` — 1 (sample std, pandas default) or 0 (population std). Any
  non-negative int is accepted.

## Design decisions

### 1. Shifted cumsum trick — O(n), numerically stable

Math for a window of size `w`:

```math
Var(x) = sum((x_j - mu)²) / (w - ddof)
       = (sum(x_j²) - sum(x_j)² / w) / (w - ddof)
```

Two cumsums (`cumsum(x)`, `cumsum(x²)`) give both running sums in O(n).

**The stability problem:** the identity `sum(x²) - sum(x)²/w` is
mathematically exact but suffers catastrophic cancellation when both
terms are large. For S&P-500-price-scale inputs (~4000), you lose ~7
digits of precision.

**The fix — shift by `x[0]`:** variance is shift-invariant. Define
`y = x - x[0]`; compute the variance formula on `y` instead of `x`.
Because `y` values are small (bounded by the range of `x` within its
observed history), both cumulative sums stay small, cancellation is
minimal, and we recover ~1e-10 to ~1e-12 precision even on
price-scale inputs.

`x[0]` is a reasonable "typical value" for financial time series and
requires only one small host-transfer to read on GPU.

### 2. Strict-window NaN policy

Same convention as `rollmean`: any NaN in the window produces NaN
output for that window. Implementation: parallel cumsum on the NaN
indicator; windows with count > 0 get NaN.

### 3. `ddof` semantics match numpy / pandas

- `ddof=1` (default): sample std — divide `ssq` by `w - 1`
- `ddof=0`: population std — divide by `w`
- `w - ddof <= 0`: return all NaN (no degrees of freedom)

### 4. `w=1` edge cases

- `w=1, ddof=0` → each window has one element = its own mean, std = 0
- `w=1, ddof=1` → `w - ddof = 0` → all NaN (matches pandas)

### 5. First-element-NaN safeguard

If `x[0]` is NaN, the shift falls back to 0 (still correct, just
slightly less optimal precision). Never cascades NaN through the whole
result.

### 6. Guard tiny negatives from FP rounding

`sum(x²) - sum(x)²/w` can be a tiny negative number for near-constant
windows due to floating-point rounding. We `xp.maximum(ssq, 0)` before
the square root to prevent NaN from a legitimate zero-variance case.

## Edge cases

| Condition | Output |
| --- | --- |
| `window == 1, ddof == 0` | all zeros |
| `window == 1, ddof == 1` | all NaN (no d.o.f.) |
| `window == len(x)` | first `n-1` NaN, last is overall std |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `ddof < 0` or non-int | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| All-NaN input | all NaN output |
| Constants in window | 0.0 (exactly, via the `maximum(., 0)` guard) |
| `x[0]` is NaN | shift falls back to 0; downstream math still correct |

## Cross-check tests

- `test_matches_pandas_uniform` — small values, `atol=1e-12` (bit-close)
- `test_matches_pandas_large_magnitude` — price-scale (~4000), `atol=1e-8`
- `test_matches_pandas_with_nans` — 5% scattered NaNs
- `test_shift_invariance` — `rollstd(x) == rollstd(x + 1000)` to 1e-9

## Test coverage (27 tests)

Golden (4 hand-computed), pandas reference (uniform + NaNs + large
magnitude), edge cases (window bounds, ddof bounds, dtype, first-NaN,
isolated NaN, all NaN, list input), property tests (non-negative,
zero-on-constants, first w-1 NaN, length preserved, shift-invariance),
CPU==GPU parity (clean + NaN).

## Direct usage in kuant

- Rolling volatility metric on M9's returns stream
- Denominator of `zscore` (once shipped)
- Bollinger-band bounds (mean ± N·std)

## Related kernels

- `kuant.stats.rollmean` — sibling; same cumsum trick + NaN policy
- `kuant.stats.zscore` — composes rollmean and rollstd (future)

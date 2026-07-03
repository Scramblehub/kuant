# zscore — Rolling window z-score

## Purpose

`zscore(x, w)[i] = (x[i] - rollmean(x, w)[i]) / rollstd(x, w, ddof)[i]`

Rolling standardization. Answers "how many rolling standard deviations
is the current value from its trailing-window mean?" The workhorse for
mean-reversion signals, anomaly detection, and any threshold logic that
needs to be scale-invariant.

## Public API

```python
from kuant.stats import zscore

result = zscore(x, window, ddof=1)
```

Same signature contract as `rollmean` / `rollstd`.

## Design decisions

### 1. Composition — first "kernel of kernels" in kuant.stats

zscore doesn't do its own cumsum work; it delegates to `rollmean` and
`rollstd`. The math is one line:

```python
z = (x - rmean) / rstd
```

Every invariant (backend/dtype/NaN/shape) inherits from the composed
kernels. Any bug fix in `rollmean` or `rollstd` immediately benefits
zscore for free. Any drift in either breaks `zscore` tests first —
composition as validation.

### 2. Trailing-window semantics — `x[i]` is the LAST value in the window

The window ending at index `i` is `x[i-w+1 : i+1]`. So `x[i]` is the
rightmost value, and its z-score is computed against the mean/std of
that trailing window. For an arithmetic progression this gives a
constant positive z (the newest value is always above the trailing
mean), not zero.

Contrast: some "centered" rolling implementations use `x[i-w/2]` as
the reference. kuant uses trailing throughout for causality (no
lookahead) — matches financial time-series convention.

### 3. Zero-std policy — NaN

Constant windows have `rollstd == 0`. Division by zero would give
`inf`; we explicitly substitute `NaN` (matches division-by-zero as
"undefined").

Implementation detail: to avoid a `RuntimeWarning` on the divide, we
mask zero-std cells to `1.0` for the division, then mask them back to
NaN.

### 4. Non-negative-std guard is implicit

`rollstd` already guarantees `>= 0` via `xp.maximum(ssq, 0)`. We use
`rstd > 0` (strict) as the mask, so both zero and NaN produce NaN
in the output.

### 5. Backend/dtype match `rollmean`'s output

Since `rollmean` and `rollstd` are called first and validate the input,
we trust their result's backend/dtype and align on them. No redundant
validation.

## Edge cases

| Condition | Output |
| --- | --- |
| First `w-1` indices | NaN (partial window, inherited) |
| Window contains any NaN | NaN (inherited from rollmean/rollstd) |
| Window is constant (rstd == 0) | NaN (explicit guard) |
| `window <= 0`, `ddof < 0` | raises `ValueError` (from rollstd) |
| `x.ndim != 1` | raises `ValueError` (from rollmean) |
| Empty input | empty output |
| All-constant input | all NaN |
| Int input | promoted to float64 |

## Cross-check tests

- `test_matches_pandas_uniform` — 500 random samples, atol=1e-10
- `test_matches_pandas_with_nans` — 5% NaNs, atol=1e-10
- `test_matches_pandas_price_scale` — large-magnitude, atol=1e-6
- `test_shift_invariance` — `zscore(x) == zscore(x + c)` for any constant c
- `test_scale_invariance` — `zscore(x) == zscore(k·x)` for positive k

Shift and scale invariance are properties of the mathematical
z-score. If either fails, the composition is broken somewhere.

## Test coverage (19 tests)

Golden (2 hand-computed), pandas reference (uniform + NaNs + price
scale), zero-std policy (constant window, all-constant), composition
invariants (first w-1 NaN, length, shift/scale invariance, dtype,
list input, raises on bad shape/window), CPU==GPU parity (clean +
constant window).

## Direct usage in kuant

- **Mean-reversion signals** — enter when `|z| > 2` (2σ deviation)
- **Anomaly gates** — flag `|z| > 3` as outliers
- **Momentum bands** — track z-score of returns to detect regime shifts
- Foundation for the V8 hourly-tilt overlay (referenced in current
  research pipeline as "z-score flip" reversal detection)

## Related kernels

- `kuant.stats.rollmean` — the numerator centering
- `kuant.stats.rollstd` — the denominator scaling

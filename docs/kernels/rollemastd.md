# rollemastd — Exponentially weighted standard deviation

## Purpose

Rolling std computed via the same exponentially-weighted recurrence as
`rollema`, but on the squared deviations. Matches
`pandas.Series.ewm(..., adjust=False).std(bias=bias)` bit-for-bit.

## Public API

```python
from kuant.stats import rollemastd
result = rollemastd(x, span=None, alpha=None, bias=False)
```

Exactly one of `span` or `alpha` must be provided; same as `rollema`.

`bias=False` (default) returns the debiased estimator matching
pandas' default. `bias=True` returns the biased estimator (no
correction) — useful when combining rollemastd results in a wider
compound estimator.

## Design decisions

### Two coupled EMA recurrences

```math
m1[i] = α · x[i]  + (1 - α) · m1[i-1]         (mean)
m2[i] = α · x[i]² + (1 - α) · m2[i-1]         (2nd moment)

var_biased[i] = max(m2[i] - m1[i]², 0)
```

Both recurrences use `scipy.signal.lfilter` for compiled-C speed,
with the same `zi = (1-α) · x[0]` initial condition trick as `rollema`.

### Debias correction (pandas-compatible)

For `adjust=False`, the weights on `x[0..k-1]` at step `k` sum to 1;
their squares sum to:

```math
Σw² = α² · (1 - β^(2(k-1))) / (1 - β²) + β^(2(k-1))
```

Unbiased variance is then `var_biased / (1 - Σw²)`. Getting this
factor right is what took several iterations — the naive
`(1 - β^(2k)) / (1 - β²)` formula is for the `adjust=True`
convention and produces visibly wrong numbers vs pandas.

### First value is NaN in the debiased path

At `k=1`, `Σw² = 1`, so the debias denominator is zero. Matches
pandas' behavior (variance of a single sample is undefined).

### GPU path — CPU fallback

Same story as `rollema`: transfer to numpy for the lfilter and the
exact Σw² formula, transfer back. A native GPU implementation would
need a parallel prefix scan.

## Cross-check tests

- Matches `pandas.ewm(alpha=α, adjust=False).std(bias=False)` on 500
  random points, `atol=1e-10`
- Same for `bias=True`

## Test coverage (3 tests)

Pandas reference (bias=False and bias=True), input validation.

## Related kernels

- `kuant.stats.rollema` — the mean recurrence this builds on
- `kuant.stats.rollstd` — window-based std counterpart
- **Future**: `kuant.stats.rollemavar` — same math, no sqrt

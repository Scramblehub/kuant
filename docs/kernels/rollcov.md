# rollcov — Rolling covariance

## Purpose

`rollcov(x, y, w, ddof=1)[i]` = sample covariance between `x[i-w+1:i+1]`
and `y[i-w+1:i+1]`.

## Public API

```python
from kuant.stats import rollcov
result = rollcov(x, y, window, ddof=1)
```

## Design decisions

### Same cumsum + shift pattern as rollcorr

Three cumsums (`sum_x`, `sum_y`, `sum_xy`), shifted by `x[0]` / `y[0]`
for stability. Formula:

```math
cov = (sum_xy - sum_x · sum_y / w) / (w - ddof)
```

Covariance is shift-invariant, so shifting both series before the
cumsums doesn't change the result but keeps magnitudes small.

### Strict-window NaN policy — union mask

If either `x` or `y` has a NaN in the window, `cov` is NaN. Same
`isnan(x) | isnan(y)` pattern as rollcorr.

### `w - ddof <= 0` returns all NaN

No degrees of freedom left for an unbiased estimator.

## Cross-check tests

- Matches `pandas.rolling().cov(other)` on uniform + NaN samples
- Symmetric: `cov(x, y) == cov(y, x)`
- `cov(x, x) == var(x)` — internal consistency with rollstd
- Length-mismatch raises

## Test coverage (5 tests)

Pandas reference (uniform + NaN), symmetry, cov-of-x-with-itself
equals variance, error handling.

## Related kernels

- `kuant.stats.rollcorr` — normalized version (divides by std_x·std_y)
- `kuant.stats.rollbeta` — cov / var(x)
- `kuant.stats.rollidio` — residual std after regressing y on x

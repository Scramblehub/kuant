# rollbeta — Rolling regression coefficient (β of y on x)

## Purpose

`rollbeta(x, y, w)[i]` = slope of the OLS regression `y ~ α + β·x`
fit over the trailing window.

```math
β = cov(x, y) / var(x)
```

Fundamental for CAPM factor exposures, pairs-trading hedge ratios,
and any rolling-linear-model application.

## Public API

```python
from kuant.stats import rollbeta
result = rollbeta(x, y, window)
```

**Argument order matters:** `x` is the independent / explanatory
variable, `y` is the dependent / response variable. Swapping them
computes `β_yx` (regression of x on y), not `β_xy`.

## Design decisions

### Direct cumsum trick — no explicit rollcov + rollvar composition

We compute `sum_x`, `sum_y`, `sum_xy`, `sum_x²` via four cumsums, then
combine algebraically:

```math
cov_num  = sum_xy - sum_x · sum_y / w
varx_num = sum_x² - sum_x² / w
β        = cov_num / varx_num
```

The `1/(w-ddof)` normalization cancels in the ratio, so `rollbeta`
has no `ddof` parameter.

### Shifted cumsums for stability

Same shift-by-first-value trick as rollcov / rollcorr.

### Zero-variance guard

If `var(x) == 0` (constant `x` in window), β is undefined → NaN.

### Union-NaN mask

NaN in either series invalidates the window (strict).

## Cross-check tests

- Perfect line: `y = 2x → β = 2` recovered exactly
- Matches `pandas.cov(x,y) / pandas.var(x)` on random samples
- Constant-x windows produce NaN

## Test coverage (3 tests)

Analytical (perfect line), pandas reference, constant-x handling.

## Related kernels

- `kuant.stats.rollcov` — the numerator
- `kuant.stats.rollstd` — sqrt of the denominator's variance
- `kuant.stats.rollidio` — residual std after regressing y on x

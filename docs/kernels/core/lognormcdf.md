# lognormcdf — Numerically stable log of Gaussian CDF

## Purpose

Compute `log(Φ(x))` without underflow in the tails.

Naive `log(normcdf(x))` underflows to `-inf` for `x < -37` in float64
because `normcdf(x)` rounds to zero.

## Public API

```python
from kuant.core import lognormcdf

result = lognormcdf(x)
```

Returns:
- `log(Φ(x))` — always finite (unless `x = nan`, in which case returns `nan`)
- `x → -∞` gives `~ -x²/2` (asymptotic behavior)
- `x → +∞` gives `→ 0`

## Design decisions

### Three-branch computation

At each element:

- **x ≥ 0**: `log1p(-normcdf(-x))` — stable because `Φ(-x)` is small
  and `log1p` handles it precisely
- **-37 ≤ x < 0**: `log(normcdf(x))` — direct, `Φ(x)` is representable
- **x < -37**: Mills asymptotic
  ```
  log Φ(x) ≈ -x²/2 - 0.5·log(2π) - log(-x) + log(1 - 1/x²)
  ```
  Error `O(1/x^4)` — better than 1e-8 relative for x < -37.

### NaN propagation

Placeholder values guard inactive branches against `log(0)`. The final
step overrides `NaN` inputs so `lognormcdf(nan) = nan`.

### Warning suppression

Inactive branches may still evaluate `log(0)` or `log1p(-1)` in
vectorized code even though `where` masks them out. We wrap the branch
computations in `np.errstate(divide='ignore', invalid='ignore')` to
keep the API clean.

## Uses

- EVT tail probability calculations (kuant.stats risk cluster)
- Extreme-quantile option pricing
- Log-likelihood computations with tail observations
- Numerically stable Bayesian model comparison

## Related

- `kuant.core.normcdf` — the CDF being log-transformed
- `kuant.core.lognormccdf` — upper-tail companion, `log(1 - Φ(x))`
- `scipy.stats.norm.logcdf` — reference implementation

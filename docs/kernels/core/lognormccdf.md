# lognormccdf — Numerically stable log of Gaussian complementary CDF

## Purpose

Compute `log(1 - Φ(x))` = `log(Φ(-x))` = `lognormcdf(-x)` without
underflow in the upper tail.

Naive `log(1 - normcdf(x))` underflows to `-inf` for `x > 37` in
float64 because `1 - normcdf(x)` rounds to zero.

## Public API

```python
from kuant.core import lognormccdf

result = lognormccdf(x)
```

Returns:
- `log(1 - Φ(x))` — always finite
- `x → +∞` gives `~ -x²/2`
- `x → -∞` gives `→ 0`

## Design decisions

### Wrapper on lognormcdf

`lognormccdf(x) = lognormcdf(-x)` by the identity
`1 - Φ(x) = Φ(-x)`. All numerical stability comes from lognormcdf.

### When to use which

- `lognormcdf(x)` — probability of being BELOW threshold x
- `lognormccdf(x)` — probability of being ABOVE threshold x

For tail-loss / VaR computations you typically want `lognormccdf`
of a large positive threshold.

## Related

- `kuant.core.lognormcdf` — the underlying primitive
- `scipy.stats.norm.logsf` — reference implementation (called "log
  survival function" in scipy)

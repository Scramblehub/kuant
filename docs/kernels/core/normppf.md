# normppf — Inverse standard-normal CDF (percent-point function)

## Purpose

Given probability `p` in `(0, 1)`, return `x` such that `Φ(x) = p`.

Inverse of `normcdf`. Foundational for quantile-based work:

- VaR / expected shortfall calculations
- Confidence intervals
- Delta → strike inversion for option chains
- Gaussian sampling via inverse-transform

## Public API

```python
from kuant.core import normppf

x = normppf(p)
```

Accepts scalar or array. Returns:
- `x` for `0 < p < 1`
- `-inf` for `p = 0`
- `+inf` for `p = 1`
- `nan` for `p ∉ [0, 1]` or `p = nan`

## Accuracy

Peter Acklam's rational approximation (2004). Maximum relative error
in `|Φ⁻¹(p)|` is ~1.15e-9, so:
- Central p: absolute error ~1e-9
- Deep tail p (e.g. p=0.001, |Φ⁻¹|≈3.1): absolute error ~4e-9

For high-precision needs (options quantile calibration), this is
usually sufficient. If tighter accuracy matters, wrap with a
Newton polish step against `normcdf`.

## Design decisions

### Three-region rational approximation

The domain splits into three regions:
- **Central** (0.02425 ≤ p ≤ 0.97575): `(p - 0.5)²` polynomial ratio
- **Lower tail** (p < 0.02425): `sqrt(-2 log p)` polynomial ratio
- **Upper tail** (p > 0.97575): same formula on `(1-p)`, then negated

Each element is evaluated in all three branches with safe placeholder
values in inactive branches; the correct branch is selected via
`xp.where`. No Python-level branching → full vectorization.

### Float64 output

Coefficients require double precision. Integer or float32 input is
promoted to float64.

### Boundary sentinels via masks

`p = 0 → -inf`, `p = 1 → +inf`, out-of-range → NaN. These sentinels
are applied after the branch computation via masked overwrites.

## Related

- `kuant.core.normcdf` — the CDF this inverts
- `kuant.core.normpdf` — density
- `scipy.stats.norm.ppf` — reference implementation

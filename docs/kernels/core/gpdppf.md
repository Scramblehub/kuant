# gpdppf — Generalized Pareto inverse CDF

## Purpose

Return `x` such that `gpdcdf(x; ξ, σ) = p`:

```math
x(p; \xi, \sigma) = \begin{cases}
  \dfrac{\sigma}{\xi}\left[(1-p)^{-\xi} - 1\right] & \xi \ne 0 \\[1em]
  -\sigma \log(1-p) & \xi = 0
\end{cases}
```

## Public API

```python
from kuant.core import gpdppf

x = gpdppf(p, xi, scale)
```

Boundary conventions:
- `p = 0` → 0
- `p = 1` → `+inf` (if `ξ ≥ 0`) or upper support `-σ/ξ` (if `ξ < 0`)
- `p ∉ [0, 1]` → `nan`
- `scale ≤ 0` → `nan`

## Uses

- Quantile-based extreme-value estimation ("what x has 1-in-100 chance")
- VaR / expected-shortfall estimation from a fitted GPD tail
- Bootstrap sampling from a GPD tail via inverse-transform

## Related

- `kuant.core.gpdcdf`, `kuant.core.gpdpdf`
- `scipy.stats.genpareto.ppf`

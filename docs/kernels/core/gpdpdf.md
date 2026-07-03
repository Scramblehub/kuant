# gpdpdf — Generalized Pareto probability density

## Purpose

PDF of the Generalized Pareto Distribution (GPD):

```math
f(x; \xi, \sigma) = \begin{cases}
  \dfrac{1}{\sigma}\left(1 + \xi \dfrac{x}{\sigma}\right)^{-\tfrac{1}{\xi} - 1}
  & \xi \ne 0 \\[1em]
  \dfrac{1}{\sigma} e^{-x/\sigma}
  & \xi = 0
\end{cases}
```

**Support:**
- `x ≥ 0` for `ξ ≥ 0`
- `0 ≤ x ≤ -σ/ξ` for `ξ < 0`

Density is 0 outside the support.

## Why GPD

The GPD is the LIMITING distribution of exceedances above a high
threshold (Pickands-Balkema-de Haan theorem 1974). This makes it
central to **Peaks-Over-Threshold (POT)** tail modeling:

1. Pick a threshold `u` in your data
2. Fit GPD to `(observations − u)` for observations above `u`
3. `ξ` estimates the tail index (heavy/exponential/bounded)
4. Extrapolate to tail probabilities beyond the data

## Public API

```python
from kuant.core import gpdpdf

f = gpdpdf(x, xi, scale)
```

- `x` — value or array
- `xi` (ξ) — shape parameter; any real number
- `scale` (σ) — must be > 0

All three broadcast to a common shape.

## Parameter meaning

| ξ | Tail behavior | Notes |
|---|---------------|-------|
| ξ > 0 | Heavy Pareto tail | Variance infinite when ξ ≥ 0.5 |
| ξ = 0 | Exponential tail | Limiting case (routed to exp formula for ξ within 1e-8 of 0) |
| ξ < 0 | Bounded tail | Hard upper support at -σ/ξ |

## Design decisions

### ξ = 0 branch is separate

For `|ξ| < 1e-8` we route to the exponential formula to avoid
numerical instability in `(-1/ξ - 1)` power. Continuous handover:
the two formulas agree in the limit.

### Out-of-support is exactly 0

Both below (x < 0) and above (x > -σ/ξ when ξ < 0) returns 0.
No NaN, no negative values.

## Related

- `kuant.core.gpdcdf`, `kuant.core.gpdppf` — CDF and inverse CDF
- Hill estimator (see `kuant.stats`) — semi-parametric tail-index
  estimator, complementary to GPD MLE
- `scipy.stats.genpareto.pdf` — reference implementation

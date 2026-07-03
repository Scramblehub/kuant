# gpdcdf — Generalized Pareto CDF

## Purpose

CDF of the Generalized Pareto Distribution:

```math
F(x; \xi, \sigma) = \begin{cases}
  1 - \left(1 + \xi \dfrac{x}{\sigma}\right)^{-1/\xi} & \xi \ne 0 \\[1em]
  1 - e^{-x/\sigma} & \xi = 0
\end{cases}
```

## Public API

```python
from kuant.core import gpdcdf

p = gpdcdf(x, xi, scale)
```

Returns:
- `0` for `x < 0`
- `F(x)` in `(0, 1)` inside support
- `1` above upper support bound (only relevant when `ξ < 0`)

## Uses

- POT modeling: convert exceedance magnitudes to probabilities
- Goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling)
  after GPD parameter estimation
- Simulation checks (round-trip through gpdppf)

## Design decisions

Same ξ = 0 handling as `gpdpdf` — routes to the exponential formula
for `|ξ| < 1e-8`.

Boundary behavior clamps to `[0, 1]` outside support to match scipy's
`genpareto.cdf`.

## Related

- `kuant.core.gpdpdf`, `kuant.core.gpdppf`
- `scipy.stats.genpareto.cdf`

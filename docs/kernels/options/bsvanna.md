# bsvanna — Black-Scholes vanna

## Purpose

Cross-partial: `∂Delta/∂σ = ∂Vega/∂S`.

```math
\text{vanna} = -e^{-qT} \cdot \varphi(d_1) \cdot \frac{d_2}{\sigma}
```

Put-call symmetric.

## Public API

```python
from kuant.options import bsvanna

vanna = bsvanna(S, K, T, r, sigma, q=0.0)
```

Units: per unit σ (decimal).

## Uses

- Hedge drift when IV changes.
- Skew risk for delta-hedged options portfolios.
- Vanna-Volga pricing corrections for FX options.

## Related

- `bscalldelta`, `bsputdelta`, `bsvega` — the first-order Greeks vanna
  bridges
- `bsvolga` — second σ-derivative

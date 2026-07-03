# bsvolga — Black-Scholes volga (vomma)

## Purpose

Second σ-derivative: `∂²Price/∂σ² = ∂Vega/∂σ`.

```math
\text{volga} = \text{Vega} \cdot \frac{d_1 \, d_2}{\sigma}
             = S \cdot e^{-qT} \cdot \varphi(d_1) \cdot \sqrt{T}
               \cdot \frac{d_1 \, d_2}{\sigma}
```

Put-call symmetric.

## Public API

```python
from kuant.options import bsvolga

volga = bsvolga(S, K, T, r, sigma, q=0.0)
```

Units: per unit σ².

## Uses

- Vol-of-vol exposure sizing.
- Pricing corrections for options on options and variance products.
- Sign flips near ATM — a diagnostic for where vega is maximized.

## Related

- `bsvega` — first σ-derivative
- `bsvanna` — cross-partial (∂Vega/∂S)

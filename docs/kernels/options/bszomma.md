# bszomma — Black-Scholes zomma

## Purpose

Third-order cross: `∂Gamma/∂σ = ∂³Price/∂S²∂σ`.

```math
\text{zomma} = \text{Gamma} \cdot \frac{d_1 \, d_2 - 1}{\sigma}
```

Put-call symmetric.

## Public API

```python
from kuant.options import bszomma

zomma = bszomma(S, K, T, r, sigma, q=0.0)
```

Units: per unit σ.

## Uses

- Gamma rebalancing when IV shifts across the vol surface.
- Sensitivity of pin-risk to vol regime.

## Related

- `bsgamma` — second-order spot derivative
- `bsspeed` — third spot derivative
- `bsvanna` — cross for delta

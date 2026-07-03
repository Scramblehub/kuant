# bsspeed — Black-Scholes speed

## Purpose

Third spot-derivative: `∂Gamma/∂S = ∂³Price/∂S³`.

```math
\text{speed} = -\frac{\text{Gamma}}{S} \cdot
               \left( \frac{d_1}{\sigma \sqrt{T}} + 1 \right)
```

Put-call symmetric.

## Public API

```python
from kuant.options import bsspeed

speed = bsspeed(S, K, T, r, sigma, q=0.0)
```

Units: per unit spot.

## Uses

- Convexity of gamma across strikes — how gamma changes as spot drifts.
- Pin-risk sizing near expiry — speed blows up as T → 0.
- Third-order delta hedging where second-order isn't enough.

## Related

- `bsgamma` — second-order spot derivative
- `bszomma` — cross with σ (∂Gamma/∂σ)

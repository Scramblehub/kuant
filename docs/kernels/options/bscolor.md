# bscolor — Black-Scholes color

## Purpose

Third-order cross: `∂Gamma/∂T = ∂³Price/∂S²∂T`.

```math
\text{color} = -\frac{e^{-qT} \varphi(d_1)}{2 S T \sigma \sqrt{T}}
               \left[ 2qT + 1 +
                       \frac{(2(r-q)T - d_2 \sigma \sqrt{T}) \, d_1}
                            {\sigma \sqrt{T}} \right]
```

Put-call symmetric.

## Sign convention

Returned as `∂Gamma/∂T` (T = time-to-expiry increasing). A positive
value means gamma grows with more time-to-expiry (typical OTM);
negative means gamma is decaying with more time (typical ATM
near-expiry).

To convert to "gamma decay per year of calendar time" flip the sign.

## Public API

```python
from kuant.options import bscolor

color = bscolor(S, K, T, r, sigma, q=0.0)
```

Units: per year. Divide by 252 for per-trading-day.

## Uses

- Gamma stability over holding periods.
- Weekend gap-risk on gamma-hedged positions.
- Near-expiry pin-risk timing.

## Related

- `bsgamma` — the derivative color differentiates
- `bszomma` — cross with σ
- `bscallcharm`, `bsputcharm` — analogous for delta

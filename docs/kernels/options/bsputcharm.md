# bsputcharm — Black-Scholes European put charm

## Purpose

Rate at which put delta bleeds as calendar time passes:

```math
\text{charm}_{\text{put}} = -\frac{\partial \Delta_{\text{put}}}{\partial T}
```

Closed form:

```math
\text{charm}_{\text{put}}
  = -q \cdot e^{-qT} \cdot \Phi(-d_1)
    - e^{-qT} \cdot \varphi(d_1) \cdot
      \frac{2(r-q)T - d_2 \sigma \sqrt{T}}{2 T \sigma \sqrt{T}}
```

## Public API

```python
from kuant.options import bsputcharm

charm = bsputcharm(S, K, T, r, sigma, q=0.0)
```

Sign convention matches `bscallcharm`. Per year.

## Design decisions

Edge cases:

- `T = 0`: returns 0.
- `S = 0`: put delta = -e^(-q·T), so charm = -q·e^(-q·T).
- `K = 0`: put worthless, delta = 0, charm = 0.
- Zero vol: returns 0.

## Related

- `bsputdelta`
- `bscallcharm`
- `bsputtheta`

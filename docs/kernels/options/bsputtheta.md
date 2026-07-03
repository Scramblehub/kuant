# bsputtheta — Black-Scholes European put theta

## Purpose

Time decay of a European put under Black-Scholes:

```math
\Theta_{\text{put}}
  = -\frac{S \cdot e^{-qT} \cdot \varphi(d_1) \cdot \sigma}{2\sqrt{T}}
    + r \cdot K \cdot e^{-rT} \cdot \Phi(-d_2)
    - q \cdot S \cdot e^{-qT} \cdot \Phi(-d_1)
```

## Public API

```python
from kuant.options import bsputtheta

theta = bsputtheta(S, K, T, r, sigma, q=0.0)
```

- Units: per **year**. Divide by 252 for per-trading-day theta,
  or by 365 for per-calendar-day theta.
- Sign: typically negative (long put loses value with time). Can be
  positive for deep ITM European puts with high interest rates —
  see the `test_deep_ITM_put_can_be_positive` case.

## Design decisions

Same structure as `bscalltheta` — analytic pass on safe placeholders,
edge cells overridden with `xp.where`:

- `T = 0` (expired) → 0
- `S = 0` → `r·K·e^(-r·T)` (put is worth `K·e^(-r·T)`, decays with time)
- `K = 0` → 0 (put worthless)
- Zero vol with T > 0 → deterministic exercise carry

## Put-call parity check

```
theta_call - theta_put = q·S·e^(-q·T) - r·K·e^(-r·T)
```

Enforced by `test_put_call_parity` in `test_bscalltheta.py`.

## Related

- `bscalltheta` — European call theta
- `bsput` — the price this theta differentiates
- `bsputdelta`, `bsgamma`, `bsvega`, `bsputrho` — other put Greeks

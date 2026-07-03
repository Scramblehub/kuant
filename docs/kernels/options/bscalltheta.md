# bscalltheta — Black-Scholes European call theta

## Purpose

Time decay of a European call under Black-Scholes:

```math
\Theta_{\text{call}}
  = -\frac{S \cdot e^{-qT} \cdot \varphi(d_1) \cdot \sigma}{2\sqrt{T}}
    - r \cdot K \cdot e^{-rT} \cdot \Phi(d_2)
    + q \cdot S \cdot e^{-qT} \cdot \Phi(d_1)
```

## Public API

```python
from kuant.options import bscalltheta

theta = bscalltheta(S, K, T, r, sigma, q=0.0)
```

- Units: per **year**. Divide by 252 for per-trading-day theta,
  or by 365 for per-calendar-day theta.
- Sign: typically negative (long call loses value with time). Can be
  positive for deep ITM European calls with high dividends.

## Design decisions

### Uniform analytic pass then edge overrides

Same pattern as the other BS Greeks. Compute the closed-form theta
on placeholder-safe inputs, then `xp.where` in edge-cell values for
S=0, K=0, T=0, and zero-vol.

### Zero-vol case

At `sigma = 0` the option payoff is deterministic. If the forward-
break-even test says the call always exercises, theta reflects the
carry of the exercise cost:

```
theta_zv = -r·K·e^(-r·T) + q·S·e^(-q·T)     if S·e^(-q·T) > K·e^(-r·T)
         =  0                                 otherwise
```

### K=0 edge

If K=0 the call is worth `S·e^(-q·T)`. Its theta is `q·S·e^(-q·T)`.

### S=0 edge

Worthless option regardless of T. Theta returns 0.

## Related

- `bsputtheta` — European put theta (put-call parity available)
- `bscall` — the price this theta differentiates
- `bscalldelta`, `bsgamma`, `bsvega`, `bscallrho` — other Greeks

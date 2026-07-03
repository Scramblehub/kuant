# bscallcharm — Black-Scholes European call charm

## Purpose

Rate at which call delta bleeds as calendar time passes:

```math
\text{charm}_{\text{call}} = -\frac{\partial \Delta_{\text{call}}}{\partial T}
```

Closed form:

```math
\text{charm}_{\text{call}}
  = q \cdot e^{-qT} \cdot \Phi(d_1)
    - e^{-qT} \cdot \varphi(d_1) \cdot
      \frac{2(r-q)T - d_2 \sigma \sqrt{T}}{2 T \sigma \sqrt{T}}
```

## Public API

```python
from kuant.options import bscallcharm

charm = bscallcharm(S, K, T, r, sigma, q=0.0)
```

- Sign: charm = -∂Delta/∂T. Positive charm means delta is bleeding
  (decreasing as time passes forward). Negative means delta is growing.
- Units: per year. Divide by 252 for per-trading-day.

## Uses

- **Hedge drift** over holding periods — how much delta hedging needs
  to change over N days given time decay alone.
- **Weekend gap risk** on delta-neutral portfolios — 3 calendar days of
  time passing = ~3·(charm/365) delta drift.
- **Pin risk** near expiry — as T→0 the ATM charm blows up.

## Put-call parity

```
charm_call - charm_put = q·e^(-q·T)
```

(Derived from delta parity `delta_call - delta_put = e^(-qT)` and the
sign convention `charm = -dDelta/dT`.)

## Design decisions

Same shape as other Greeks: uniform analytic pass on safe placeholders,
edge cells overridden via `xp.where`.

- `T = 0`: charm undefined (delta is a step); returns 0.
- `S = 0`: delta = 0 always; charm = 0.
- `K = 0`: call = S·e^(-qT), delta = e^(-qT), so charm = -q·e^(-qT).
- Zero vol away from forward-parity: delta constant; charm = 0.

## Related

- `bscalldelta` — the delta this charm differentiates
- `bsputcharm` — put charm
- `bscalltheta` — time decay of price (not delta)

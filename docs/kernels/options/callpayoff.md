# callpayoff — European call expiry payoff

## Purpose

Actual exercise value at expiry: `max(S - K, 0)`.

Not a Greek — this is the payoff function. Foundational building block
for spread pricers, Monte Carlo terminal values, and any expiry-time
analytic.

## Public API

```python
from kuant.options import callpayoff

payoff = callpayoff(S, K)
```

Broadcasts S and K; backend follows either cupy input.

## Design decisions

### Trivial by design

Just `xp.maximum(S - K, 0)` with the boilerplate: backend detection,
dtype resolution, and int→float64 promotion so integer strikes/spots
work without surprise.

### int → float64 promotion

Prevents silent bugs where users pass strike prices as int and get
integer division downstream.

### Backend-preserving

numpy in → numpy out, cupy in → cupy out. Batched over any shape.

## Related

- `putpayoff` — European put payoff
- `bscall` — the risk-neutral present value that discounts this payoff
- Building block for future spread pricers (vertical spreads, straddles)

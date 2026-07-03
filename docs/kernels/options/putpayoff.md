# putpayoff — European put expiry payoff

## Purpose

Actual exercise value at expiry: `max(K - S, 0)`.

## Public API

```python
from kuant.options import putpayoff

payoff = putpayoff(S, K)
```

## Intrinsic parity

```
callpayoff(S, K) - putpayoff(S, K) == S - K
```

Enforced by `test_intrinsic_parity` in the test suite.

## Related

- `callpayoff` — European call payoff
- `bsput` — the risk-neutral present value that discounts this payoff

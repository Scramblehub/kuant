# bscalldelta — Black-Scholes European call delta

## Purpose

Sensitivity of call price to underlying:

```math
delta = ∂C/∂S = e^(-q·T) · Φ(d1)
```

Range: `[0, 1]`. Opposite in both sign and Φ argument from put delta:

| | Formula | Range |
| --- | --- | --- |
| `bsputdelta` | `-e^(-q·T) · Φ(-d1)` | `[-1, 0]` |
| `bscalldelta` | `+e^(-q·T) · Φ(d1)` | `[0, 1]` |

## Public API

```python
from kuant.core import bscalldelta
d = bscalldelta(S, K, T, r, sigma, q=0.0)
```

## Design decisions

### 1. Uses `_bs_common.prepare_bs` for setup

Same pattern as bsputdelta. One line of math:
`xp.exp(-c.q * c.T_safe) * normcdf(c.d1)`.

### 2. S=0/K=0 answers swap versus put delta

A call at S=0 is worthless (delta=0), while a put at S=0 is at guaranteed
exercise (delta=-e^(-q·T)). Same swap for K=0. See `bscall.md` for the
put-call asymmetry discussion.

## Edge cases

| Condition | Call delta |
| --- | --- |
| Normal | analytic |
| T=0, S > K | 1 |
| T=0, S ≤ K | 0 |
| σ=0, forward > K | e^(-q·T) |
| σ=0, forward ≤ K | 0 |
| S=0 | 0 (call worthless) |
| K=0 | e^(-q·T) (guaranteed exercise) |
| NaN | NaN |

## Cross-check tests

- `test_put_call_parity_for_delta` — machine-precision identity
  `delta_call - delta_put = e^(-q·T)` on 500 random points, `atol=1e-13`
- `test_delta_matches_finite_difference` — `delta ≈ (bscall(S+h) - bscall(S-h)) / (2h)`

## Test coverage (21 tests)

Golden values (6 scipy-derived, including deep ITM/OTM boundaries), 1000-
point scipy reference, put-call parity for delta (500 points), range [0, 1],
monotonic in S, FD vs bscall, edge cases + dtype + GPU parity.

## Related kernels

- `kuant.core.bscall` — bscalldelta is its dS-derivative
- `kuant.core.bsputdelta` — parity partner
- `kuant.core.normcdf` — called once per delta element
- `kuant.core.bsgamma` — d(bscalldelta)/dS, put-call symmetric (same as put's)
- `kuant.core.bsvega` — sibling first-order Greek, put-call symmetric

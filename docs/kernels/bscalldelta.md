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

## Put-call parity for delta

```math
delta_call - delta_put = e^(-q·T)
```

Machine-precision identity — the test `test_put_call_parity_for_delta`
checks 500 random points to `atol=1e-13`. If either kernel drifts, this test
catches it.

## Public API

```python
from kuant.core import bscalldelta
d = bscalldelta(S, K, T, r, sigma, q=0.0)
```

## Edge cases (mirror bsputdelta with swapped S=0/K=0)

| Condition | Call delta |
| --- | --- |
| Normal | analytic |
| T=0, S>K | 1 |
| T=0, S≤K | 0 |
| σ=0, forward>K | e^(-q·T) |
| σ=0, forward≤K | 0 |
| **S=0** | **0** (call worthless) |
| **K=0** | **e^(-q·T)** (guaranteed exercise) |
| NaN | NaN |

The S=0/K=0 answers are swapped versus put delta — see `bscall.md` for the
put-call asymmetry discussion.

## Design decisions

Same pattern as `bsputdelta`. Uses `_bs_common.prepare_bs`. One line of
math: `xp.exp(-c.q * c.T_safe) * normcdf(c.d1)`.

## Cross-check tests (21 tests)

- Golden values (6 scipy-derived, including deep ITM/OTM boundaries)
- 1000-point scipy reference match
- **Put-call parity for delta** (500 random points)
- Range [0, 1], monotonic in S
- FD vs bscall
- Edge cases + dtype + GPU parity

## Related kernels

- `kuant.core.bscall` — bscalldelta is its dS-derivative
- `kuant.core.bsputdelta` — parity partner
- `kuant.core.bsgamma` — gamma is bscalldelta's dS-derivative (same as put's)

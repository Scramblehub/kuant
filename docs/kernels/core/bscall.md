# bscall — Black-Scholes European call pricer

## Purpose

Prices a European call on a dividend-paying stock:

```math
d1 = [ln(S/K) + (r - q + σ²/2) · T] / (σ · √T)
d2 = d1 - σ · √T
call = S · e^(-q·T) · Φ(d1) - K · e^(-r·T) · Φ(d2)
```

Uses `kuant.core.normcdf` twice per element. Related to `bsput` by
put-call parity: `C - P = S · e^(-q·T) - K · e^(-r·T)`.

## Public API

```python
from kuant.core import bscall
c = bscall(S, K, T, r, sigma, q=0.0)
```

Signature identical to `bsput`. Returns non-negative values.

## Design decisions

### 1. Uses `_bs_common.prepare_bs` for setup

Same 20 lines of setup as bsput. See `_bs_common.py` for the flow.

### 2. Sign differences from bsput

Both formulas use `d1` and `d2`, but the `Φ` arguments and outer signs
flip:

| Kernel | Formula |
| --- | --- |
| `bsput` | `+K·e^(-r·T)·Φ(-d2) - S·e^(-q·T)·Φ(-d1)` |
| `bscall` | `+S·e^(-q·T)·Φ(d1)  - K·e^(-r·T)·Φ(d2)` |

`bsput` uses `Φ(-d1), Φ(-d2)`; `bscall` uses `Φ(d1), Φ(d2)`. `bsput`
weights the strike-discount term with `+`; `bscall` weights the
spot-discount term with `+`.

### 3. Edge case order matters — Case 4 must come last

```python
# 1. Normal path (analytic formula)
# 2. Deterministic (T<=0 or sigma<=0) - intrinsic discounted
# 3. K == 0 (guaranteed exercise, unlimited payoff) - overrides (2)
# 4. S == 0 (worthless) - overrides (2) and (3)
```

Case 4 has to come last because K=0 AND S=0 → call is worthless, not
infinite. When both are zero, "worthless" wins.

## Edge cases

| Condition | Call |
| --- | --- |
| Normal | analytic |
| T=0 (expired) | max(S-K, 0) |
| σ=0, T>0 | max(S·e^(-q·T) - K·e^(-r·T), 0) |
| S=0 | 0 (worthless) |
| K=0 | S·e^(-q·T) (guaranteed exercise) |
| NaN | NaN |

The **S=0 / K=0 answers swap** between put and call — a put worth K when
S=0 becomes a call worth 0, and vice versa. Cleanest place in the whole
BS family where put-call asymmetry appears in code.

## Cross-check tests

- `test_put_call_parity_atm` — hard identity, checked to 1e-12
- `test_put_call_parity_random` — 1000 random points, `atol=1e-12`
- `test_matches_reference_uniform` — 1000 random points vs scipy directly

Put-call parity is a mathematical identity, not a numerical approximation.
If `bscall - bsput ≠ S·e^(-q·T) - K·e^(-r·T)` at machine precision, one of
the two kernels has a bug.

## Test coverage (24 tests)

Golden values (5 scipy-derived), scipy reference (1000 random), put-call
parity (deterministic + 1000 random), broadcasting, edge cases (T=0
ITM/OTM/ATM, σ=0 with/without exercise, S=0, K=0, NaN, float32), property
tests (non-neg, monotonic in strike / vol, bounded by S·e^(-q·T)),
CPU==GPU parity + backend promotion.

## Direct usage in kuant

Not currently used by the V8 stack (M9 is put-only). Present for
completeness — future work on the call side (bull-call spreads, covered
calls, IV surface calibration) will lean on this.

## Related kernels

- `kuant.core.bsput` — put price, parity partner
- `kuant.core.normcdf` — called twice per bscall element
- `kuant.core.bscalldelta` — dC/dS, uses same normcdf pattern
- `kuant.core.bscallrho` — dC/dr, uses same normcdf pattern
- `kuant.core.bsgamma`, `kuant.core.bsvega` — put-call symmetric, shared with bsput

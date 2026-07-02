# bscall — Black-Scholes European call pricer

## Purpose

Prices a European call on a dividend-paying stock:

```math
d1 = [ln(S/K) + (r - q + σ²/2) · T] / (σ · √T)
d2 = d1 - σ · √T
call = S · e^(-q·T) · Φ(d1) - K · e^(-r·T) · Φ(d2)
```

Uses `kuant.core.normcdf` twice per element.

## Public API

```python
from kuant.core import bscall
c = bscall(S, K, T, r, sigma, q=0.0)
```

Signature identical to `bsput`. Returns non-negative values.

## Sign differences from bsput

Both formulas use `d1` and `d2`, but the `Φ` arguments and outer signs flip:

| Kernel | Formula |
| --- | --- |
| `bsput` | `+K·e^(-r·T)·Φ(-d2) - S·e^(-q·T)·Φ(-d1)` |
| `bscall` | `+S·e^(-q·T)·Φ(d1)  - K·e^(-r·T)·Φ(d2)` |

`bsput` uses `Φ(-d1), Φ(-d2)`; `bscall` uses `Φ(d1), Φ(d2)`.
`bsput` weights the strike-discount term with +; `bscall` weights the
spot-discount term with +.

## Put-call parity — the strongest cross-check

```math
C - P = S · e^(-q·T) - K · e^(-r·T)
```

This is a hard mathematical identity — not a numerical approximation. If
`bscall - bsput ≠ S·e^(-q·T) - K·e^(-r·T)` at machine precision, one of the
two kernels has a bug.

The test `test_put_call_parity_random` checks 1000 random points and passes
to `atol=1e-12`. Machine epsilon on doubles is 2.22e-16, so we're within
~5000x machine precision — the residual is accumulated floating-point
rounding, not algorithmic error.

## Design decisions

### 1. Reuses `_bs_common.prepare_bs`

Same 20 lines of setup as bsput (backend detection, dtype policy,
broadcasting, NaN init, d1/d2/normal precomputation). See `_bs_common.py`
docstring for details.

### 2. Edge cases mirror bsput's structure but flip direction

| Condition | Put | Call |
| --- | --- | --- |
| Normal | analytic | analytic |
| T=0 (expired) | max(K-S, 0) | max(S-K, 0) |
| σ=0, T>0 | max(K·e^(-r·T) - S·e^(-q·T), 0) | max(S·e^(-q·T) - K·e^(-r·T), 0) |
| S=0 | K·e^(-r·T) (exercise) | 0 (worthless) |
| K=0 | 0 (worthless) | S·e^(-q·T) (exercise) |
| NaN | NaN | NaN |

The **S=0 / K=0 answers swap** between put and call — a put worth K when
S=0 becomes a call worth 0, and vice versa. This is the cleanest place to
see put-call asymmetry in the edge handling.

### 3. Layered edge-case order matters

```python
# 1. Normal path (analytic formula)
# 2. Deterministic (T<=0 or sigma<=0) - intrinsic discounted
# 3. K == 0 (guaranteed exercise, unlimited payoff) - overrides (2)
# 4. S == 0 (worthless) - overrides (2) and (3)
```

Case 4 has to come last because K=0 AND S=0 → call is worthless, not
infinite. When both are zero, "worthless" wins.

## Test coverage (24 tests)

1. **Golden values** — 5 scipy-derived cases: ATM, ITM, OTM, dividend, Hull textbook
2. **Reference match** — 1000 random parameter sets vs scipy directly
3. **Put-call parity** — deterministic ATM check + 1000 random points to 1e-12
4. **Broadcasting** — strike curve
5. **Edge cases** — T=0 (ITM/OTM/ATM), σ=0 (exercise/no-exercise), S=0, K=0,
   NaN, float32 preservation
6. **Property tests** — non-neg, monotonic in strike (decreasing), monotonic
   in vol (positive vega), bounded above by S·e^(-q·T)
7. **CPU==GPU parity + backend promotion**

## Direct usage in kuant

Not currently used by the V8 stack (M9 is put-only). Included for
completeness — future work on the call side (bull-call spreads, covered
calls, IV surface calibration) will lean on this.

## Performance notes

- CPU: composed of two `scipy.special.ndtr` calls per element (~100 ns/element)
- GPU: two `cupyx.scipy.special.erf` calls + a handful of ops (~2-3 ns/element on RTX 4090)
- Break-even vs CPU: ~50k elements

## Related kernels

- `kuant.core.bsput` — put price. Related by put-call parity.
- `kuant.core.normcdf` — called twice per bscall element
- `kuant.core.bscalldelta` — `+e^(-q·T) · Φ(d1)`, range [0, 1]
- `kuant.core.bscallrho` — `+T · K · e^(-r·T) · Φ(d2)`, range [0, +∞)
- `kuant.core.bsgamma`, `kuant.core.bsvega` — put-call symmetric, shared with bsput

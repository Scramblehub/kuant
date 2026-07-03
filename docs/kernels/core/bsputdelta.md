# bsputdelta — Black-Scholes European put delta

## Purpose

Rate of change of a European put's price with respect to the underlying:

```math
delta = ∂P/∂S = -e^(-q·T) · Φ(-d1)
```

Range: `[-1, 0]`. Direct powering of the M9 TP monitor's D70 rule.

## Public API

```python
from kuant.core import bsputdelta

d = bsputdelta(S, K, T, r, sigma, q=0.0)
```

Signature identical to `bsput`.

## Design decisions

### 1. Uses `_bs_common.prepare_bs` for setup

Shared with all BS kernels — backend detection, broadcasting, dtype
policy, NaN init, and d1/d2/normal mask. See `_bs_common.py` for the flow.
Each Greek is ~20 lines of formula on top of the shared helper.

### 2. One line of actual math

```python
delta = -xp.exp(-q * T) * normcdf(-d1)
```

Everything else in the file is edge-case handling. This is what "kernel
composition" looks like at its cleanest: one small mathematical fact on
top of a lower-level primitive.

### 3. Convention at K == S, T = 0

The kink of `max(K-S, 0)` at K=S is not differentiable — left derivative
is -1, right is 0. We pick the **right derivative** (0), matching Hull.
Matters for automated exercise decisions at expiry.

### 4. σ=0 case is a step function, not a curve

Physically real. No vol → deterministic forward → put either always
exercises or never does. The step at `K = S·e^((r-q)·T)` is not a
numerical artifact.

## Edge cases

| Condition | Delta |
| --- | --- |
| Normal | analytic |
| T=0 (expired), K > S | -1 |
| T=0 (expired), K ≤ S | 0 |
| σ=0, K·e^(-r·T) > S·e^(-q·T) | -e^(-q·T) |
| σ=0, K·e^(-r·T) ≤ S·e^(-q·T) | 0 |
| S=0 | -e^(-q·T) (guaranteed exercise) |
| K=0 | 0 |
| NaN | NaN |

## Cross-check tests

- `test_delta_matches_finite_difference` — `delta ≈ (bsput(S+h) - bsput(S-h)) / (2h)`
  with `h=1e-4`, agrees to ~1e-8

The strongest check in the file. If either bsputdelta or bsput drifts, this
fails. Bumping through the price kernel proves the analytic derivative is
consistent with the price it should be derived from.

## Test coverage (24 tests)

Golden values (6 scipy-derived), 1000-point scipy reference, broadcasting
(strike curve), edge cases (T=0 ITM/OTM/ATM, σ=0 exercise/no-exercise/with
div, S=0, K=0, NaN, float32), property tests (bounded [-1, 0], monotonic
in strike and spot, finite-difference agreement), CPU==GPU parity.

## Direct usage in kuant

**M9 TP monitor D70 rule.** Close the put when `|delta| ≥ 0.70`:

```python
delta = bsputdelta(spot, strike, tenor_years, r, iv)
if abs(delta) >= 0.70:
    close_put()
```

At `|delta| ≥ 0.70`, trailing past the strike is often worse than the
loss from being called away. A one-line D70 rule can materially improve
put-monitoring on a real position book.

## Related kernels

- `kuant.core.bsput` — bsputdelta is its dS-derivative
- `kuant.core.normcdf` — called once per delta element
- `kuant.core.bscalldelta` — parity partner: `delta_call - delta_put = e^(-q·T)`
- `kuant.core.bsgamma` — d(bsputdelta)/dS, put-call symmetric
- `kuant.core.bsvega` — dP/dσ, put-call symmetric

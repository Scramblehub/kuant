# bscallrho — Black-Scholes European call rho

## Purpose

Sensitivity of call price to risk-free rate:

```math
rho = ∂C/∂r = T · K · e^(-r·T) · Φ(d2)
```

Range: `[0, +∞)`. Opposite sign from put rho, uses `Φ(d2)` not `Φ(-d2)`:

| | Formula | Range |
| --- | --- | --- |
| `bsputrho` | `-T · K · e^(-r·T) · Φ(-d2)` | `(-∞, 0]` |
| `bscallrho` | `+T · K · e^(-r·T) · Φ(d2)` | `[0, +∞)` |

**Sign intuition:** higher rates *discount* the strike more, which *helps*
call holders (they pay less in present-value terms) and *hurts* put holders
(the K they'd receive is worth less). Same magnitude, opposite signs.

## Public API

```python
from kuant.core import bscallrho
r_ = bscallrho(S, K, T, r, sigma, q=0.0)
```

For "rho per 1% change in rate", divide by 100.

## Design decisions

### 1. Uses `_bs_common.prepare_bs` for setup

Same shared helper as every BS kernel.

### 2. Non-monotonicity in strike — a subtle correctness note

Unlike put rho (monotonically decreasing in K), call rho is **NOT
monotonic in K**. My first property test got this wrong.

- Deep ITM (K << S): `Φ(d2) → 1`, so `rho → T·K·e^(-r·T)`, growing linearly in K
- Deep OTM (K >> S): `Φ(d2) → 0`, so `rho → 0`

The product `K · Φ(d2)` peaks somewhere between and drops off both ways.

Call rho **is** monotonic in S — that's the correct property test.

### 3. Both S=0 and K=0 give rho=0

Different from put rho, where S=0 gives nonzero (`-T·K·e^(-r·T)`, the
guaranteed-exercise case). Reason: a call at K=0 is worth `S·e^(-q·T)`,
which doesn't depend on r; a put at S=0 is worth `K·e^(-r·T)`, which does.

## Edge cases

| Condition | Call rho |
| --- | --- |
| Normal | analytic |
| T=0 (expired) | 0 |
| σ=0, S·e^(-q·T) > K·e^(-r·T) | T·K·e^(-r·T) |
| σ=0, S·e^(-q·T) ≤ K·e^(-r·T) | 0 |
| S=0 | 0 (call worthless) |
| K=0 | 0 (call worth S·e^(-q·T), no r dependence) |
| NaN | NaN |

## Cross-check tests

- `test_put_call_parity_for_rho` — `rho_call - rho_put = T · K · e^(-r·T)`
  on 500 random points, `atol=1e-10`
- `test_rho_matches_finite_difference_of_price` — bumps `r` by ±1e-5 in
  bscall, agrees with analytic rho to ~1e-6

## Test coverage (19 tests)

Golden values, scipy reference, put-call parity, non-negative,
S-monotonicity (not K), FD vs bscall, edge cases, dtype, GPU parity.

## Direct usage in kuant

Same story as bsputrho — least-used Greek for retail short-dated options.
Present for completeness.

## Related kernels

- `kuant.core.bscall` — bscallrho is its dr-derivative
- `kuant.core.bsputrho` — parity partner (opposite sign)
- `kuant.core.normcdf` — called once per rho element

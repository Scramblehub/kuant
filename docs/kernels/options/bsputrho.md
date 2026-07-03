# bsputrho — Black-Scholes European put rho

## Purpose

Sensitivity of put price to the risk-free rate:

```math
rho = ∂P/∂r = -T · K · e^(-r·T) · Φ(-d2)
```

Put-specific. Call rho has opposite sign and uses `Φ(d2)` — see
`bscallrho`.

Differences from the other put Greeks:

- Uses `T` (not `√T`) — from differentiating `e^(-r·T)`
- Uses `Φ(-d2)` (not `φ(d1)`) — composes on `normcdf`, not `normpdf`
- Always `≤ 0` — higher rates discount the strike more, lowering put value

## Public API

```python
from kuant.core import bsputrho
r_ = bsputrho(S, K, T, r, sigma, q=0.0)
```

For "rho per 1% change in rate", divide by 100.

## Design decisions

### 1. Uses `_bs_common.prepare_bs` for setup

Same shared helper as every BS kernel. Composition point: `normcdf` (like
bsput and bsputdelta), NOT normpdf.

### 2. σ=0 and S=0 edges give NON-ZERO rho

Unlike gamma/vega where all edges are 0. Reason: at S=0 the put is worth
`K·e^(-r·T)`, which still depends on r → `∂/∂r = -T·K·e^(-r·T)`. Same for
σ=0 when the put exercises in the deterministic case.

### 3. σ=0 branch decides via forward inequality

```python
K * exp(-r*T) > S * exp(-q*T)  →  put exercises  →  rho = -T*K*exp(-r*T)
else                            →  put worthless →  rho = 0
```

Consistent with the σ=0 branches in bsputdelta.

## Edge cases

| Condition | Rho |
| --- | --- |
| Normal | analytic |
| T=0 (expired) | 0 (no future to discount) |
| σ=0, K·e^(-r·T) > S·e^(-q·T) | -T·K·e^(-r·T) |
| σ=0, K·e^(-r·T) ≤ S·e^(-q·T) | 0 |
| S=0 | -T·K·e^(-r·T) (guaranteed exercise) |
| K=0 | 0 |
| NaN | NaN |

## Cross-check tests

- `test_rho_matches_finite_difference_of_price` — bumps `r` by ±1e-5 in
  bsput and confirms `bsputrho ≈ d(bsput)/dr` to ~1e-6

FD tests need smaller `h` for rho (1e-5) than for delta (1e-4) — rate is
small in absolute terms (0.05 typical), so bumps have to shrink
proportionally to stay in the linear regime.

## Test coverage (18 tests)

Golden values (6 including ATM/OTM/ITM/dividend/deep ITM/deep OTM), scipy
match on 1000 uniform, non-positive property, monotonic in strike, FD
cross-check vs bsput, edge cases (T=0, σ=0 exercise + worthless, S=0,
K=0, NaN, float32), CPU==GPU parity.

## Direct usage in kuant

Least-used Greek in practice for short-dated retail options (rates move
slowly relative to a 7-DTE M9 put's tenor). Present for completeness —
useful for rho-adjusted P&L attribution if the M9 sleeve is back-tested
across a rate-shock period like 2022.

## Related kernels

- `kuant.core.bsput` — bsputrho is its dr-derivative
- `kuant.core.normcdf` — called once per rho element
- `kuant.core.bscallrho` — parity partner, opposite sign, uses Φ(d2) not Φ(-d2)

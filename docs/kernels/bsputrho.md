# bsputrho — Black-Scholes European put rho

## Purpose

Sensitivity of put price to the risk-free rate:

```math
rho = ∂P/∂r = -T · K · e^(-r·T) · Φ(-d2)
```

Note the differences from the other put Greeks:

- Uses `T` (not `√T`) — comes from differentiating `e^(-r·T)`
- Uses `Φ(-d2)` (not `φ(d1)`) — composes on `normcdf`, not `normpdf`
- Always `≤ 0` — higher rates discount the strike more, lowering put value

## Public API

```python
from kuant.core import bsputrho
r_ = bsputrho(S, K, T, r, sigma, q=0.0)
```

Same signature as `bsput`. Returns values in `(-∞, 0]`.

For "rho per 1% change in rate", divide by 100.

## Call rho (for reference)

Call rho has **opposite sign** and uses `Φ(d2)` not `Φ(-d2)`:

```math
rho_call = +T · K · e^(-r·T) · Φ(d2)
```

So this kernel is put-specific — do NOT reuse for calls. When we ship
`bscallrho` in the future, that will be its own file.

## Design decisions

### 1. Same overall pattern as the other put Greeks

- Backend detection, broadcasting, dtype (defaulted-q trick), NaN via
  `full_like(nan)`, uniform-compute-then-mask.
- Composition point: `normcdf` (like bsput and bsputdelta), NOT normpdf.

### 2. Edge cases — mostly zero, one exception

| Condition | Rho |
| --- | --- |
| Normal | analytic |
| T=0 | 0 (no future to discount) |
| σ=0, exercises | -T·K·e^(-r·T) |
| σ=0, worthless | 0 |
| S=0 | -T·K·e^(-r·T) (guaranteed exercise) |
| K=0 | 0 |
| NaN | NaN |

The `σ=0` and `S=0` edges have **non-zero** rho — unlike gamma/vega. Makes
sense: at S=0 the put is definitely worth `K·e^(-r·T)`, which *does* depend
on r. So `∂/∂r = -T·K·e^(-r·T)`.

### 3. σ=0 case decides based on same forward inequality as delta

```python
K * exp(-r*T) > S * exp(-q*T)  →  put exercises  →  rho = -T*K*exp(-r*T)
else                            →  put worthless →  rho = 0
```

Consistent with the σ=0 branches in bsputdelta.

## Finite-difference test

`test_rho_matches_finite_difference_of_price` bumps `r` by ±1e-5 in bsput
and confirms `bsputrho == d(bsput)/dr` to ~1e-6. Same cross-check pattern
we used for delta/gamma/vega — proves the derivative chain is coherent.

## Test coverage (18 tests)

- 6 golden values (ATM, OTM, ITM, dividend, deep ITM, deep OTM)
- scipy match on 1000 uniform
- non-positive property, monotonic in strike
- FD cross-check vs bsput
- Edge cases: T=0, σ=0 (both branches), S=0, K=0, NaN, float32
- CPU==GPU parity

## Real-world usage

Rho is the least-used Greek in practice for short-dated retail options
(rates move slowly relative to a 7-DTE M9 put's tenor). Included for
completeness — if the M9 sleeve is ever back-tested across a rate-shock
period like 2022, you'll want rho-adjusted P&L attribution.

## Related kernels

- `kuant.core.bsput` — bsputrho is its dr-derivative
- `kuant.core.normcdf` — called once per rho element
- **Future**: `kuant.core.bscallrho` — opposite sign, different Φ argument

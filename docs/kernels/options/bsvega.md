# bsvega — Black-Scholes vega (calls and puts)

## Purpose

Sensitivity of price to implied volatility:

```math
vega = ∂P/∂σ = S · e^(-q·T) · φ(d1) · √T
```

**Put-call symmetric**: same value for a call and a put with identical
inputs. One kernel, both directions.

Range: `[0, +∞)`. Peaks near ATM at longer tenors.

**Units convention.** Vega here is `∂P/∂σ` with `σ` in **decimal** (e.g.
0.20 = 20% IV). For "vega per 1% change in IV", divide by 100. Some
libraries pre-divide by 100 — kuant does not, so callers get the
mathematically clean derivative.

## Public API

```python
from kuant.core import bsvega
v = bsvega(S, K, T, r, sigma, q=0.0)
```

## Design decisions

### 1. Uses `_bs_common.prepare_bs` for setup

Shared with all BS kernels.

### 2. Same put-call-symmetry pattern as bsgamma

One kernel for both directions. Enforces the symmetry in code.

### 3. Composition on `normpdf`, uses `d1` only

Same pattern as `bsgamma`.

### 4. All edges collapse to zero

Vega vanishes at expiry (no future vol to matter), at zero vol (limit),
at zero spot (no S dependence to price).

## Edge cases

| Condition | Vega |
| --- | --- |
| Normal | analytic |
| T=0 | 0 |
| σ=0 | 0 |
| S=0 | 0 |
| K=0 | 0 |
| NaN | NaN |

## Cross-check tests

- `test_vega_matches_finite_difference_of_price` —
  `bsvega ≈ (bsput(S, K, T, r, σ+h) - bsput(S, K, T, r, σ-h)) / (2h)`

## Test coverage (15 tests)

Golden, scipy reference, non-neg, ATM peak, FD-cross-check vs price,
edge cases, dtype, GPU parity.

## Direct usage in kuant

**Implied-vol solvers.** Newton-Raphson iteration:

```math
σ_new = σ_old - (price(σ_old) - target) / vega(σ_old)
```

Central to any IV surface work. Also useful for M9's vol-shock scenarios:
`portfolio_vega × vol_move` estimates first-order P&L from an IV spike.

## Related kernels

- `kuant.core.normpdf` — called once per vega element
- `kuant.core.bsput`, `kuant.core.bscall` — vega is their dσ-derivative
- `kuant.core.bsputdelta`, `kuant.core.bscalldelta` — sibling first-order Greeks
- `kuant.core.bsgamma` — shares the put-call-symmetric pattern

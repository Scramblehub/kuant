# impvol — Implied volatility solver (vectorized Newton-Raphson)

## Purpose

Given a market option price and its (S, K, T, r, q), find the sigma that
Black-Scholes says produces that price:

```math
bsput(S, K, T, r, σ, q) == price     (put)
bscall(S, K, T, r, σ, q) == price    (call)
```

Direct application: real-time IV inversion for open M9 puts and any
future IV surface work.

## Public API

```python
from kuant.options import impvol

sigma = impvol(price, S, K, T, r, is_call=False, q=0.0,
               tol=1e-8, max_iter=100)
```

- All numeric inputs support broadcasting.
- `is_call` is a boolean flag (broadcasts to all elements).
- Returns NaN where the input is arbitrage-violating or the solver fails.

## Design decisions

### 1. Vectorized Newton, not per-element loop

All elements iterate in parallel. Converged elements are frozen via a
mask; the loop terminates when every in-bounds element is within `tol`
of its target. Typical convergence: 3-5 iterations for well-behaved
inputs.

Trade-off: `xp.all(converged)` requires a small host sync each iteration
on GPU. Cheap compared to the vectorized bs* calls that dominate cost.

### 2. Manaster-Koehler initial guess

```math
σ₀ = √(|ln(S/K) + r·T| · 2 / T)
```

More robust across moneyness than the ATM-optimal Brenner-Subrahmanyam.
Works from deep OTM to deep ITM without special casing.

Clamped to `[1e-6, 10.0]` immediately.

### 3. No-arbitrage bounds check

Prices outside the theoretical bounds get NaN without wasting iteration:

| Direction | Bounds |
|---|---|
| Put | `[max(K·e^(-r·T) - S·e^(-q·T), 0), K·e^(-r·T)]` |
| Call | `[max(S·e^(-q·T) - K·e^(-r·T), 0), S·e^(-q·T)]` |

Also NaN: `T ≤ 0`, `S ≤ 0`, `K ≤ 0`.

### 4. Vega-zero guard

`vega < 1e-8` → skip the step for that element (Newton is useless when
the curve is flat). If it never gets a usable vega, the final validation
returns NaN for it.

### 5. Sigma clamping every iteration

`σ ← clip(σ - step, 1e-6, 10.0)`. Prevents Newton overshoot into
non-physical vol ranges (negative or > 1000% annualized).

### 6. Final validation gate

After max_iter (or early break), we recompute the price at the final
sigma and check `|residual| < 10 · tol`. Elements that pass AND were
in arbitrage bounds get their sigma; everyone else gets NaN.

The `10 · tol` factor is defensive — Newton's last step may have moved
sigma slightly past the exact root, so we widen a touch to accept
answers that are "close enough".

## Numerical caveats

### Low-vega regime

For deep OTM / deep ITM options with low vol, the price surface is very
flat. Both Newton and bisection (scipy's brentq) hit a numerical floor
where sigma can shift by 1e-4 without moving price by more than 1e-10.
`impvol` returns *a* valid sigma, but "the" implied vol isn't well
defined in that regime.

Practically: filter your inputs to `vega > 0.1` before treating the
output as a precise number. Below that, take the answer as an
approximate ballpark.

### Short-tenor + low-vol combo

Same issue as above; tenor and vol both suppress vega together. `1/365`
tenor at 10% vol is fine; `1/365` at 3% vol is problematic.

## Edge cases

| Condition | Output |
|---|---|
| Price < no-arb lower bound | NaN |
| Price > no-arb upper bound | NaN |
| `T <= 0` or `S <= 0` or `K <= 0` | NaN |
| Vega always below threshold | NaN |
| Newton diverges → sigma clamped, residual too big | NaN |
| Scalar input | Python float output |
| Batched | array output with elementwise NaN |

## Cross-check tests

- **Round-trip**: `bsput(σ) → impvol → σ_recovered`, tolerance 1e-6 in
  the well-behaved region
- **scipy.optimize.brentq**: independent bisection solver on 50 random
  parameter tuples. Newton and brentq agree to 1e-5 where vega > 0.1
- **Batched vs scalar**: batched impvol gives same answer as per-element
  scalar loop

## Test coverage (43 tests)

- 21 round-trip tests (7 sigmas × 3 (S,K,T,r,q) tuples) for puts
- 9 round-trip tests for calls
- 2 scipy.optimize.brentq random-parameter matches (put and call)
- 4 no-arbitrage-violation tests
- 5 edge cases (short tenor, high vol, deep OTM)
- 2 batched tests (puts and calls)
- Scalar-in / scalar-out
- dtype preservation (float32)
- CPU==GPU parity (round-trip + backend)

## Direct usage in kuant

**M9 TP monitor.** Current inline BS pricer computes theoretical price
from an assumed sigma. Replacing with `impvol` inverts the observed
market price into the implied vol, which is the actual live risk metric
for the D70 delta rule and any vega-based position sizing.

## Related kernels

- `kuant.core.bsput`, `kuant.core.bscall` — the pricers we invert
- `kuant.core.bsvega` — the Newton step's derivative
- **Future**: `kuant.options.impvol_bisection` — pure-bisection fallback
  for the low-vega tail

# bsgamma — Black-Scholes gamma (calls and puts)

## Purpose

Second derivative of price with respect to spot:

```math
gamma = ∂²P/∂S² = e^(-q·T) · φ(d1) / (S · σ · √T)
```

**Put-call symmetric**: the value is the same for a call and a put with
identical inputs. That's why the kernel name has no put/call prefix — a
single function serves both directions.

Range: `[0, +∞)`. Peaks near ATM at short tenors; drops off in ITM/OTM.

## Public API

```python
from kuant.core import bsgamma
g = bsgamma(S, K, T, r, sigma, q=0.0)
```

Same signature as `bsput` / `bscall`.

## Direct usage in kuant

Position risk metric. `gamma × spot_move²` estimates a next-day P&L jump
from a directional move — useful in the M9 monitor when a held put's
underlying is running.

## Design decisions

### 1. One function for both directions

Put-call symmetry is a mathematical fact, not an implementation choice.
Encoding it as one kernel (rather than `bsputgamma` + `bscallgamma` aliases)
makes the fact enforceable in code and un-losable in refactors.

### 2. Composition on `normpdf`, uses `d1` only

Never needs `d2` even though `_bs_common.prepare_bs` computes it. The extra
computation is one subtraction per element — cheaper than an if-branch and
keeps the setup helper simple. Zero-cost tradeoff.

### 3. All edges collapse to zero

| Condition | Gamma |
| --- | --- |
| Normal | analytic |
| T=0, σ=0, S=0, K=0 | 0 |
| NaN | NaN |

Gamma vanishes everywhere the option's convexity vanishes. Single
`where(edge, zero, out)` mask covers all four cases.

## Cross-check tests (strongest validators)

- `bsgamma == d(bsputdelta)/dS` via bump-and-difference
- `bsgamma == d²(bsput)/dS²` via central second difference

These bump through delta and price kernels, so a drift anywhere in the
chain (bsgamma, bsputdelta, bsput, normcdf, normpdf) surfaces here first.

## Test coverage (16 tests)

Golden, scipy reference, non-neg, ATM peak, FD-cross-check vs delta,
FD²-cross-check vs price, edge cases, dtype, GPU parity.

## Related kernels

- `kuant.core.normpdf` — called once per gamma element
- `kuant.core.bsputdelta`, future `kuant.core.bscalldelta` — gamma is their
  dS-derivative
- `kuant.core.bsvega` — shares the put-call-symmetric pattern

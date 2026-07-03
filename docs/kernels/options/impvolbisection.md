# impvolbisection — Implied vol via vectorized bisection

## Purpose

Recover the sigma that reproduces an observed option price, using
bisection instead of Newton-Raphson. Complement to `impvol`.

Bisection **cannot diverge** and works fine on flat-vega tails
(deep OTM, near expiry) where Newton's step blows up.

## Public API

```python
from kuant.options import impvolbisection

sigma = impvolbisection(price, S, K, T, r, is_call=False, q=0.0,
                        tol=1e-8, max_iter=100,
                        sigma_lo=1e-6, sigma_hi=10.0)
```

Returns implied vol shaped by broadcast of inputs. NaN where:
- `price < intrinsic` (below no-arbitrage bound)
- `price > K·e^(-r·T)` (above upper bound)
- target not bracketed by `[sigma_lo, sigma_hi]`

## When to use bisection vs Newton (`impvol`)

| Consideration | Newton (`impvol`) | Bisection (`impvolbisection`) |
|---------------|-------------------|-------------------------------|
| Speed | Faster (~3-5 iterations) | Slower (~30 iterations) |
| Robustness | Diverges on flat vega | Guaranteed to bracket |
| Deep OTM near expiry | Fails | Solves |
| Batch throughput | Better | Fine |
| Correctness on pathologies | Sometimes NaN | Always converges |

**Default choice:** Newton (`impvol`) for typical solves. Fall back to
bisection when you know you'll hit flat-vega zones (options screens
with many far-OTM strikes, calibration on illiquid chains).

## Design decisions

### Vectorized bisection with per-element bracketing

All elements bisect simultaneously. Converged elements keep bisecting
but their `hi - lo` is already below `tol` so they don't move
meaningfully. Early exit via global check `all(hi - lo < tol)`.

### Bracket is [1e-6, 10.0] by default

Covers 0.0001% to 1000% annualized vol. Configurable via
`sigma_lo` / `sigma_hi` if you have prior knowledge (screening a
liquid-strike surface, e.g., [0.05, 2.0]).

### No-arbitrage check at endpoint prices

Before iterating, `price_fn(sigma_lo)` and `price_fn(sigma_hi)` are
computed. If the target price is outside `[price_lo, price_hi]`, the
solver marks NaN immediately (no wasted iterations).

### Monotonicity of price(sigma)

BS price is strictly monotone increasing in sigma for both calls and
puts. Bisection assumes this. No verification is needed at runtime.

## Related

- `kuant.options.impvol` — Newton-Raphson version, faster typical case
- `kuant.options.bsvega` — used by Newton to compute Jacobian
- `kuant.core.bscall`, `kuant.core.bsput` — the pricing kernels

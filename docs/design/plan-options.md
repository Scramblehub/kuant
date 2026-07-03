# Plan: kuant.options expansion

`kuant.options` currently has one kernel: `impvol` (vectorized
Newton-Raphson IV solver). Tomorrow's target: complete the option
analytics category so kuant covers the standard research toolkit.

## Scope

Six new kernels, split into three themes.

### Theme 1 — Time-decay Greeks (θ) and charm (∂δ/∂t)

The remaining Greeks that aren't already in `kuant.core`.

| Kernel | Formula | Location |
|---|---|---|
| `bscalltheta` | ∂C/∂t (annualized, negative for calls) | `kuant.core.bscalltheta` |
| `bsputtheta` | ∂P/∂t (annualized, negative for puts) | `kuant.core.bsputtheta` |
| `bscallcharm` | ∂δ_call/∂t | `kuant.core.bscallcharm` |
| `bsputcharm` | ∂δ_put/∂t | `kuant.core.bsputcharm` |

**These go in `kuant.core`, not `kuant.options`.** Same rationale as
existing Greeks: they're pure math primitives on `bsput`/`bscall`.

### Theme 2 — IV solver robustness

| Kernel | Purpose | Location |
|---|---|---|
| `impvolbisection` | Bisection-based IV solver as fallback for the low-vega tail where Newton diverges | `kuant.options.impvolbisection` |

Complements `impvol`. Falls back to bracketing bisection when Newton
step overshoots or vega is below a threshold. Slower per call but
guaranteed to converge on any monotonic price function.

### Theme 3 — Option chain utilities

| Kernel | Purpose | Location |
|---|---|---|
| `optionchain` | Utilities for building/filtering an option chain from a strike grid + expiries | `kuant.options.optionchain` |

Not a "kernel" in the numerical-primitive sense — more of a helper
for research pipelines. Small.

## Design decisions to lock

### Theta sign convention

Standard: negative for a long option position (time decay costs the
holder). Match Bloomberg / most textbooks. Users can flip if they
prefer positive-per-day.

### Theta units — annualized or per-day?

Textbook is annualized (∂/∂T where T is in years). Traders often
prefer per-day (annualized / 365). Support both:

```python
theta_annual = bscalltheta(S, K, T, r, sigma, q=0.0)      # default
theta_daily  = bscalltheta(S, K, T, r, sigma, q=0.0) / 365
```

Default annualized; document the /365 pattern for daily.

### `impvolbisection` — separate function or `stable=True` flag on `impvol`?

Option A: separate `kuant.options.impvolbisection(...)`. Cleaner
namespace, explicit choice.
Option B: `impvol(..., stable=False)` with `stable=True` triggering
bisection. Single-function API.

Recommend Option A. `impvolbisection` is a distinct algorithm; hiding
it behind a flag conflates two things. Reader who sees `impvol` in
code knows exactly which algorithm ran.

### `optionchain` shape

Returns a structured object (dataclass? DataFrame?) with strike/tenor
grid metadata + prices per (strike, tenor) cell. Design later — this
is more of a research helper than a numerical kernel.

## Formulas

### `bscalltheta`

```math
θ_call = -S·e^(-q·T)·φ(d1)·σ/(2√T)
       + q·S·e^(-q·T)·Φ(d1)
       - r·K·e^(-r·T)·Φ(d2)
```

### `bsputtheta`

```math
θ_put = -S·e^(-q·T)·φ(d1)·σ/(2√T)
      - q·S·e^(-q·T)·Φ(-d1)
      + r·K·e^(-r·T)·Φ(-d2)
```

### `bscallcharm` (annualized)

```math
charm_call = -e^(-q·T)·(q·Φ(d1) - φ(d1)·(2(r-q)T - d2·σ√T)/(2T·σ√T))
```

### `bsputcharm`

```math
charm_put = -e^(-q·T)·(-q·Φ(-d1) - φ(d1)·(2(r-q)T - d2·σ√T)/(2T·σ√T))
```

Verify against a reference implementation (`py_vollib`, `mibian`, or
scipy manually) before shipping.

## Implementation order

1. **`bscalltheta` + `bsputtheta`** — one file `bstheta.py` with both
   like the delta/rho pairs. Golden values from a scipy reference.
2. **`bscallcharm` + `bsputcharm`** — one file `bscharm.py`.
   Cross-check via finite-difference of `bsputdelta` / `bscalldelta`.
3. **`impvolbisection`** — sits in `kuant.options`. Bracketing
   bisection with early-exit tolerance. Cross-check against `impvol`
   in the normal regime.
4. **`optionchain`** — design + implement last. Small helper, easy
   to shape once the Greeks are all in place.

Docs go in `docs/kernels/core/` (theta, charm) and
`docs/kernels/options/` (impvolbisection, optionchain).

## Rough estimate

- Theta pair: 1 hour (formula + tests + docs)
- Charm pair: 1 hour
- `impvolbisection`: 1.5 hours (need robust bracket logic)
- `optionchain`: 1 hour if we keep it minimal

Total: 4.5 hours if all four ship. Realistic to complete tomorrow.

# digitalprice: cash-or-nothing digital (binary) option

## Purpose

Closed-form price of a European cash-or-nothing binary. Pays a fixed
`cash` at maturity if the underlying finishes in the money, zero
otherwise:

```math
digital_call = cash * exp(-r*T) * N(d2)
digital_put  = cash * exp(-r*T) * N(-d2)
d2           = (log(S/K) + (r - q - sigma^2/2)*T) / (sigma*sqrt(T))
```

`N(d2)` is the risk-neutral probability of finishing ITM; the price is
that probability discounted, times the fixed payout.

## Public API

```python
from kuant.options import digitalprice

call = digitalprice(100.0, 100.0, 1.0, 0.05, 0.20, cash=1.0)
put  = digitalprice(100.0, 100.0, 1.0, 0.05, 0.20, cash=1.0, is_call=False)
```

- `S, K, T, r, sigma, q` broadcast per standard BS convention.
- `cash` (kw-only), fixed payout on knock-in; default 1.0.
- `is_call` (kw-only), `True` for the call analogue (pays if `S_T > K`),
  `False` for the put (pays if `S_T < K`).
- Returns scalar or array, shape follows broadcast of inputs.

## Design decisions

### 1. Uses the shared `prepare_bs` context

The kernel piggybacks on `kuant.core._bs_common.prepare_bs` for
backend detection, dtype promotion, `T_safe` / `sigma_safe` guarding,
and `d1, d2` computation. Only the payout leg (`cash * disc * N(±d2)`)
is kernel-specific.

### 2. No Greeks in this release

The step-function payoff makes delta / gamma singular at the strike as
`T -> 0` (a Dirac-scaled spike). A smoothed-delta variant (small vega
kernel around the strike) may follow; not shipped in v0.6.

### 3. Backend and dtype preserved

Same contract as the rest of `kuant.options`: numpy in / numpy out,
cupy in / cupy out, integer inputs promoted to float64, float32 inputs
kept at float32.

### 4. Domain guard mirrors vanilla BS

`(T > 0) & (sigma > 0) & (S > 0) & (K > 0)` gates the returned price;
elements failing the guard fall through to `prepare_bs`'s default NaN /
intrinsic handling.

## Edge cases

| Condition | Output |
| --- | --- |
| `S >> K`, low sigma, call | approaches `cash * exp(-r*T)` |
| `S << K`, low sigma, call | approaches 0 |
| Put-call parity | `digital_call + digital_put == cash * exp(-r*T)` at machine precision |
| `T <= 0` or `sigma <= 0` | NaN via `prepare_bs` fallthrough |
| Scalar inputs | Python float |

## Cross-check tests

Tests in `tests/options/test_exotics_batch8.py::TestDigital`:

- `test_atm_gaussian_reference`, ATM call price in `(0.4, 0.6)`, matches
  `0.5*exp(-r*T)` up to the drift term.
- `test_put_call_parity`, `|call + put - cash*exp(-r*T)| < 1e-10`,
  confirms parity at machine precision.
- `test_deep_itm_call_approaches_cash`, `S=200, K=100, sigma=0.05`,
  price is within `1e-6` of `exp(-r*T)`.
- `test_deep_otm_call_near_zero`, `S=50, K=100, sigma=0.05`, price < 0.05.

## Direct usage in kuant

Building block for one-touch / no-touch structures on the M9 monitor:
priced with the same closed form after a barrier / probability
substitution. Also seeds the smoothed-delta variant if that lands.

## Related kernels

- `kuant.core.bscall`, `kuant.core.bsput`, vanilla references for parity.
- `kuant.options.gapprice`, uses the same `d1, d2` substitution with a
  differing payout strike.
- `kuant.core.normcdf`, the underlying `N(.)`.

## References

- Reiner and Rubinstein 1991, "Unscrambling the Binary Code."
- Haug 2007, "The Complete Guide to Option Pricing Formulas," ch. 4.19.

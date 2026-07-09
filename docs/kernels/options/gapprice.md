# gapprice: European gap option

## Purpose

Closed-form price of a European gap option: a vanilla whose TRIGGER
strike `K_trigger` decides whether the payoff activates, and whose
PAYOFF strike `K_payoff` sets the magnitude:

```math
gap_call = (S_T - K_payoff) * indicator(S_T > K_trigger)
gap_put  = (K_payoff - S_T) * indicator(S_T < K_trigger)
```

When `K_payoff != K_trigger` the payoff has a discontinuous jump (the
"gap") at the trigger. Closed form:

```math
d1 = (log(S/K_trigger) + (r - q + sigma^2/2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
gap_call = S*exp(-q*T)*N(d1) - K_payoff*exp(-r*T)*N(d2)
gap_put  = K_payoff*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)
```

`d1, d2` use `K_trigger` (probability of activation); discounted
`K_payoff` enters the second term.

## Public API

```python
from kuant.options import gapprice

# Vanilla-equivalent (triggers match):
v = gapprice(100.0, 100.0, 100.0, 1.0, 0.05, 0.20)

# Nonzero gap, trigger 100, payoff 90:
g = gapprice(100.0, 100.0, 90.0, 1.0, 0.05, 0.20)
```

- `S, T, r, sigma, q` broadcast as standard BS.
- `K_trigger` activates the payoff.
- `K_payoff` sets the subtracted / added strike inside the payoff.
- `is_call` (kw-only), default `True`.
- Returns scalar or array following broadcast.

## Design decisions

### 1. Reduces to vanilla at `K_trigger == K_payoff`

When triggers match, the formula collapses to the standard
Black-Scholes call / put. Verified at machine precision by
`test_triggers_match_recovers_vanilla`.

### 2. `prepare_bs` seeded with `K_trigger`

`prepare_bs(S, K_trigger, T, r, sigma, q)` produces `d1, d2` off the
trigger, then `K_payoff` is broadcast in as a separate `xp.asarray`
promoted to the shared `out_dtype`. This keeps the kernel identical to
vanilla structure with a single substitution.

### 3. Discontinuous payoff, well-behaved price

The gap payoff jumps at `K_trigger` at expiry, but the pre-expiry price
is smooth in all six BS inputs (integration over the terminal density
smooths the step). Greeks are analytic and not exposed by this kernel
in v0.6.

### 4. Domain guard mirrors vanilla BS

`(T > 0) & (sigma > 0) & (S > 0) & (K_trigger > 0)` gates the returned
price; failing elements fall through to `prepare_bs`'s default handling.

### 5. Backend and dtype preserved

Same contract as `kuant.core`, cupy / numpy / float32 / float64 handled
transparently.

## Edge cases

| Condition | Output |
| --- | --- |
| `K_payoff == K_trigger` | equals vanilla `bscall` / `bsput` |
| `K_payoff < K_trigger` (call) | pays more than vanilla, price > vanilla |
| `K_payoff > K_trigger` (call) | can be negative (holder pays into the gap) |
| `T <= 0` or `sigma <= 0` | NaN via fallthrough |
| Scalar inputs | Python float |

Negative prices are legitimate: if `K_payoff > K_trigger` in the call,
the holder must pay into the gap when the option triggers just above
`K_trigger`. Consistent with Reiner-Rubinstein.

## Cross-check tests

Tests in `tests/options/test_exotics_batch8.py::TestGap`:

- `test_triggers_match_recovers_vanilla`, matches `bscall` to `1e-10`
  when `K_trigger == K_payoff == 100`.
- `test_gap_pays_more_than_vanilla_when_payoff_below_trigger`, sanity
  check that `K_payoff=90 < K_trigger=100` lifts the call value.
- `test_put`, ATM put returns positive value.

## Direct usage in kuant

Not currently used in the M9 stack. Documented as the closed-form
counterpart to `digitalprice`; the two are the primitives that
Reiner-Rubinstein exotics decompose into.

## Related kernels

- `kuant.options.digitalprice`, the pure-binary sibling (same `d2` gate,
  no `d1` leg).
- `kuant.core.bscall`, `kuant.core.bsput`, vanilla references.
- `kuant.core.normcdf`, underlying `N(.)`.

## References

- Reiner and Rubinstein 1991, "Unscrambling the Binary Code."
- Haug 2007, "The Complete Guide to Option Pricing Formulas," ch. 4.17.

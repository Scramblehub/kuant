# lookbackprice: floating-strike lookback option

## Purpose

Closed-form price of a European floating-strike lookback under
continuous monitoring. Payoff references the extremum of the path:

```math
lookback_call = S_T - min_{0 <= t <= T} S_t     (buy at the low)
lookback_put  = max_{0 <= t <= T} S_t - S_T     (sell at the high)
```

Goldman-Sosin-Gatto 1979 closed form (Haug 2007 ch. 5.13). With
`b = r - q`:

```math
a1 = (log(S/S_extreme) + (b + sigma^2/2)*T) / (sigma*sqrt(T))
a2 = a1 - sigma*sqrt(T)
lookback_call = S*exp(-q*T)*N(a1) - S_min*exp(-r*T)*N(a2)
              + S*exp(-r*T) * (sigma^2 / (2*b)) * (
                 (S_min/S)^(2*b/sigma^2) * N(-a1 + 2*b*sqrt(T)/sigma)
                 - exp(b*T) * N(-a1)
              )
```

Symmetric formula for the put with `S_max` in place of `S_min`.

For a fresh contract with no path history, pass `S_extreme = S`.

## Public API

```python
from kuant.options import lookbackprice

# Fresh ATM lookback call, continuous monitoring:
lc = lookbackprice(100.0, 100.0, 1.0, 0.05, 0.20)
# Seasoned put with running max 110:
lp = lookbackprice(100.0, 110.0, 0.5, 0.05, 0.20, is_call=False)
```

- `S`, current spot.
- `S_extreme`, running minimum (call) or maximum (put) seen so far.
- `T, r, sigma, q` broadcast as standard BS.
- `is_call` (kw-only), default `True`.
- Returns scalar or array following broadcast.

## Design decisions

### 1. Reuses `prepare_bs` for context only

`prepare_bs(S, S_extreme, ...)` supplies the backend, `T_safe`,
`sigma_safe`, and `sqrt_T`. The standard `d1, d2` it produces are
discarded; `a1, a2` are recomputed against `S_extreme` (the running
extremum, not a strike).

### 2. Small-`b` guard around `r - q = 0`

The `sigma^2 / (2*b)` prefactor blows up as `b -> 0`. Two guards:

- Vector-level: `b_safe = where(|b| < 1e-8, 1e-8, b)` keeps the
  arithmetic finite.
- Scalar-level: on scalar inputs where `|b| < 1e-6`, a
  `KuantNumericWarning` is emitted with code `[KW-LOOKBACK-NEAR-ZERO-CARRY]`:

```text
kuant.lookbackprice: |b|=|r-q|=... is near zero; the sigma^2/(2b)
prefactor is guarded but the returned price loses precision. Prefer
the b=0 limiting form for production use.  [KW-LOOKBACK-NEAR-ZERO-CARRY]
```

Users should treat the price as indicative in that regime; the `b=0`
limiting series expansion is the correct production formula and is not
implemented in v0.6.

### 3. Domain guard

`(T > 0) & (sigma > 0) & (S > 0) & (S_extreme > 0)` gates the price.
The kernel does not enforce `S_extreme <= S` (call) or `S_extreme >= S`
(put) at the API layer; passing a violating extremum yields a formal
price that has no financial interpretation.

### 4. Continuous monitoring only

The closed form assumes the extremum is observed continuously.
Discrete-monitoring corrections (Broadie-Glasserman-Kou) are not applied;
a daily-monitored contract will be priced at an upper bound.

### 5. Backend and dtype preserved

Standard `kuant.core` contract.

## Edge cases

| Condition | Output |
| --- | --- |
| `S_extreme = S` (fresh) | positive; > vanilla ATM call / put |
| `abs(b) < 1e-6`, scalar | warning `[KW-LOOKBACK-NEAR-ZERO-CARRY]` + guarded price |
| `T <= 0` or `sigma <= 0` | NaN via fallthrough |
| Discrete monitoring | overestimated by known Broadie-Glasserman-Kou factor |
| Scalar inputs | Python float |

## Cross-check tests

Tests in `tests/options/test_exotics_batch8.py::TestLookback`:

- `test_fresh_call_positive_greater_than_vanilla`, fresh lookback call
  exceeds vanilla ATM call (matches the "always exercised optimally"
  intuition).
- `test_fresh_put_positive`, fresh lookback put is positive.
- `test_mc_agreement_within_stderr`, 50k-path 252-step Monte Carlo
  vs analytic call, tolerance `|mc - analytic| / analytic < 0.07`.
  Empirically the gap sits in the 3-6% band; discretization bias
  (finite step count) explains most of it.

## Direct usage in kuant

Not currently used in the M9 stack. Documented for future path-dependent
overlays (SMM lookback exit rules, drawdown-triggered rebalance).

## Related kernels

- `kuant.core.bscall`, `kuant.core.bsput`, vanilla references bounded
  below by the lookback prices.
- `kuant.core.normcdf`, underlying `N(.)`.

## References

- Goldman, Sosin, and Gatto 1979, "Path Dependent Options: Buy at the
  Low, Sell at the High," Journal of Finance.
- Haug 2007, "The Complete Guide to Option Pricing Formulas," ch. 5.13.

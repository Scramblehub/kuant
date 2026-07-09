# powerprice: power option (payoff `max(S^n - K, 0)`)

## Purpose

Closed-form price of a European power option: payoff is a power of the
terminal underlying against a fixed strike:

```math
power_call = max(S_T^n - K, 0)
power_put  = max(K - S_T^n, 0)
```

For `n > 1` the payoff convex-amplifies gains beyond the strike; for
`n in (0, 1)` it diminishes them; `n = 1` recovers the plain vanilla.

Heynen-Kat 1996 closed form under Black-Scholes with continuous
dividend yield `q`:

```math
mu_n = n*(r - q) + n*(n - 1) * sigma^2 / 2
d1_n = (log(S / K^(1/n)) + (r - q + (n - 1/2)*sigma^2) * T) / (sigma*sqrt(T))
d2_n = d1_n - n*sigma*sqrt(T)
power_call = S^n * exp((mu_n - r)*T) * N(d1_n) - K*exp(-r*T)*N(d2_n)
```

Standard sign flips for the put. The `exp((mu_n - r)*T)` factor
accounts for the risk-neutral drift of `S_t^n` under the standard
Black-Scholes measure.

## Public API

```python
from kuant.options import powerprice

# Squared ATM call (K = S^2 = 10000):
p2 = powerprice(100.0, 10000.0, 1.0, 0.05, 0.20, n=2.0)

# Vanilla recovered at n=1:
v = powerprice(100.0, 100.0, 1.0, 0.05, 0.20, n=1.0)
```

- `S, T, r, sigma, q` broadcast as standard BS.
- `K` compared against `S_T^n`; pick `K ~ S^n` for ATM parity.
- `n` (kw-only), power exponent, default 2.0. Must be non-zero.
- `is_call` (kw-only), default `True`.
- Returns scalar or array following broadcast.

## Design decisions

### 1. Reduces to vanilla at `n = 1`

Substituting `n = 1` collapses `mu_n` to `r - q`, `d1_n / d2_n` to
standard BS `d1 / d2`, and `S^n = S`. Verified at machine precision by
`test_n_equals_one_recovers_vanilla`.

### 2. Scalar `n` guard on zero

`n = 0` degenerates: `S^0 = 1` and the payoff is either always 0 or
always `1 - K`. The kernel raises `ValueError`:

```text
kuant.powerprice: 'n' must be non-zero (payoff S^n is undefined/degenerate at n=0).
[KE-VAL-RANGE]
```

Array-typed `n` is not batched in v0.6 (power options are typically
priced at a single exponent per contract). A scalar float is expected.

### 3. Effective strike `K^(1/n)` in `log(S/K^(1/n))`

The natural substitution: rewriting `S_T^n > K` as `S_T > K^(1/n)`
recovers a vanilla-shaped moneyness variable, with the drift and vol
scaling absorbed by `mu_n` and the `n*sigma*sqrt(T)` term in `d2_n`.

### 4. Reuses `prepare_bs` context only

Backend / dtype / `T_safe` / `sigma_safe` come from `prepare_bs`; the
standard `d1, d2` it produces are discarded and `d1_n, d2_n` are
recomputed against the effective strike.

### 5. Domain guard

`(T > 0) & (sigma > 0) & (S > 0) & (K > 0)` gates the price; failing
elements fall through to the default NaN / intrinsic path.

### 6. Backend and dtype preserved

Standard `kuant.core` contract.

## Edge cases

| Condition | Output |
| --- | --- |
| `n = 1.0` | matches `bscall` / `bsput` to `1e-10` |
| `n = 0.0` | raises `ValueError` `[KE-VAL-RANGE]` |
| `n > 1`, `K = S^n` (ATM comparable) | price scales super-linearly with `n` |
| `n in (0, 1)`, call | payoff concave, price below vanilla notional |
| `T <= 0` or `sigma <= 0` | NaN via fallthrough |
| Scalar inputs | Python float |

## Cross-check tests

Tests in `tests/options/test_exotics_batch8.py::TestPower`:

- `test_n_equals_one_recovers_vanilla`, `p1 == bscall` at `1e-10`.
- `test_squared_positive`, `n=2` ATM (K = 10000) yields price > 100.
- `test_put_positive_atm`, `n=2` ATM put positive.
- `test_higher_n_amplifies_call`, price scales up with `n` when
  `K = S^n` (comparable moneyness).

## Direct usage in kuant

Not currently used in the M9 stack. Documented as a convex-payoff
building block: candidate for future tail-boost overlays where the
squared / cubed payoff amplifies rare wins beyond the vanilla capture.

## Related kernels

- `kuant.core.bscall`, `kuant.core.bsput`, vanilla case `n = 1`.
- `kuant.options.gapprice`, `kuant.options.digitalprice`, sibling
  exotics.
- `kuant.core.normcdf`, underlying `N(.)`.

## References

- Heynen and Kat 1996, "Pricing and Hedging Power Options," Financial
  Engineering and the Japanese Markets 3(3).
- Haug 2007, "The Complete Guide to Option Pricing Formulas," ch. 4.24.

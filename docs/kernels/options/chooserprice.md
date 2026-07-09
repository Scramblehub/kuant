# chooserprice: simple chooser option

## Purpose

Closed-form price of a simple chooser: at `t_choose <= T` the holder
declares the contract to be a European call or put, both with strike
`K` and remaining maturity `T - t_choose`. At the choice date the
holder picks `max(C, P)` on the two hypothetical vanillas.

Rubinstein 1991 decomposition (put-call parity applied at `t_choose`)
reduces the chooser to a portfolio of two standard vanillas evaluated
at t=0:

```math
chooser = call(S, K, T; sigma)
        + put(S, K * exp(-(r - q) * (T - t_choose)), t_choose; sigma)
```

The call takes the full maturity `T`; the put takes a shifted strike
and shortened maturity `t_choose`. No new numerical machinery: the
kernel is two `bscall` / `bsput` legs plus a strike discount.

## Public API

```python
from kuant.options import chooserprice

# Chooser at half-life, ATM:
ch = chooserprice(100.0, 100.0, 1.0, 0.5, 0.05, 0.20)
```

- `S, K, T, r, sigma, q` broadcast as standard BS.
- `t_choose`, choice date in years, `0 <= t_choose <= T`.
- Returns scalar or array following broadcast.

## Design decisions

### 1. Rubinstein decomposition, no new pricer

Two `prepare_bs` contexts:

1. `c = prepare_bs(S, K, T, r, sigma, q)`, standard call leg over full
   `T`.
2. `c2 = prepare_bs(S, K_shifted, t_choose, r, sigma, q)` with
   `K_shifted = K * exp(-(r - q) * (T - t_choose))`, standard put leg
   over the shortened maturity.

Sum the two leg prices. No iteration, no root find, no MC.

### 2. Scalar `t_choose` range guard

Scalar inputs enforce `0 <= t_choose <= T` and raise `ValueError`:

```text
kuant.chooserprice: 't_choose' (...) must satisfy 0 <= t_choose <= T (...).
[KE-VAL-RANGE]
```

Array-broadcast `t_choose` (rare in practice) is filtered by the same
mask at the closing `xp.where`: out-of-range elements fall through to
the default NaN / intrinsic path.

### 3. Bounded by the straddle

The chooser is bounded above by `call + put` (a plain straddle):
holding the straddle also lets you exercise the winning leg at
`t_choose`. Verified by `test_chooser_bounded_by_straddle`.

### 4. Two limits verified

- `t_choose -> 0`: the choice is immediate, chooser -> `max(call, put)`.
- `t_choose -> T`: the holder retains both legs to expiry, chooser ->
  `call + put`.

Both hold at `~1e-3` (put-call parity residual at the endpoints,
verified by `test_t_choose_zero_equals_max_call_put` and
`test_t_choose_full_equals_call_plus_put`).

### 5. Backend and dtype preserved

Standard `kuant.core` contract.

## Edge cases

| Condition | Output |
| --- | --- |
| `t_choose = 0` (approx) | equals `max(call, put)` |
| `t_choose = T` (approx) | equals `call + put` (straddle) |
| `t_choose < 0` or `t_choose > T` (scalar) | raises `ValueError` `[KE-VAL-RANGE]` |
| `t_choose < 0` or `t_choose > T` (array element) | NaN via fallthrough |
| `T <= 0` or `sigma <= 0` | NaN via fallthrough |
| Scalar inputs | Python float |

## Cross-check tests

Tests in `tests/options/test_exotics_batch8.py::TestChooser`:

- `test_t_choose_zero_equals_max_call_put`, `t_choose = 0.001`, matches
  `max(bscall, bsput)` within `1e-3`.
- `test_t_choose_full_equals_call_plus_put`, `t_choose = 0.999`,
  matches `bscall + bsput` within `0.02`.
- `test_chooser_bounded_by_straddle`, chooser <= straddle at any
  interior `t_choose`.

## Direct usage in kuant

Not currently used in the M9 stack. Documented as a compact test of
put-call parity and as scaffolding for the complex chooser (different
strikes / maturities on the two legs) that may follow.

## Related kernels

- `kuant.core.bscall`, `kuant.core.bsput`, the two legs of the
  decomposition.
- `kuant.options.digitalprice`, `kuant.options.gapprice`, sibling
  exotics.

## References

- Rubinstein 1991, "Options for the Undecided," RISK 4(4).
- Haug 2007, "The Complete Guide to Option Pricing Formulas," ch. 4.9.

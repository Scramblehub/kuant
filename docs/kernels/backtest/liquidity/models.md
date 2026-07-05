# models - FillModel family

## Purpose

Encode the size-to-slippage relationship as a small family of typed
models the fill kernel can call polymorphically. A `FillModel`
answers: "given an order of `size` against today's `adv`, what
fraction of the reference price does this fill pay in slippage?" The
three shipped models cover the standard progression from a naive
constant, through a linear-in-participation form, to the concave
square-root form the empirical literature favors for large orders.

## Public API

```python
from kuant.backtest.liquidity import (
    FlatSlippage,
    LinearImpact,
    SquareRootImpact,
)
```

All three are frozen dataclasses that expose a single method:

```python
compute_slippage(size: float, adv: float, side: int) -> float
```

Returns a signed fraction of the reference price. `execute_fill` uses
the return value to compute `fill_price = price * (1 + side *
slippage_frac)`. Positive values widen the fill in the direction of
the order; buys pay more, sells receive less.

### `FlatSlippage(bps)`

Constant per-order slippage, no size dependence.

```python
FlatSlippage(bps: float)   # bps >= 0
```

Model form:

```
slippage_frac = bps / 10_000
```

### `LinearImpact(k)`

Slippage scales linearly with participation rate.

```python
LinearImpact(k: float)     # k >= 0
```

Model form:

```
slippage_frac = (k / 10_000) * (|size| / adv)
```

`k` is the slippage at 100% participation, in basis points. A `k` of
20 means an order equal to today's ADV pays 20 bps; the same
coefficient at 10% participation pays 2 bps.

### `SquareRootImpact(k)`

Almgren-Chriss square-root form.

```python
SquareRootImpact(k: float) # k >= 0
```

Model form:

```
slippage_frac = (k / 10_000) * sqrt(|size| / adv)
```

`k` is again the slippage at 100% participation. At 10%
participation the slippage is `k * sqrt(0.10) / 10_000`, roughly
`0.316 * k` bps.

## Design decisions

### 1. Three models, not one

`FlatSlippage` is the naive baseline; useful for regression tests
and toy simulations, wrong for anything at scale. `LinearImpact` is
a reasonable first-cut for mid-size orders that do not sweep the
book. `SquareRootImpact` matches the empirical literature on large
meta-orders. Shipping all three keeps the caller honest: the fill
model is a modelling choice, not a hidden default.

Behaviour of `FlatSlippage` matches the `slippage` parameter in
common backtest tooling: a small order and a huge order pay the same
haircut. This is fine only when the strategy trades in sizes small
relative to ADV; otherwise it is a source of silent alpha inflation.

### 2. Concavity of the square-root form

`SquareRootImpact` is concave in size: each additional share pays
less incremental slippage than the last. This reflects real
order-book behaviour where aggressive fills sweep the top of book
cheaply and pay exponentially more for the tail; the aggregate
slippage grows sub-linearly.

By contrast `LinearImpact` implies constant marginal slippage per
share. That is defensible up to modest fractions of ADV but
overshoots on very large orders relative to what the depth of book
would actually charge.

### 3. Coefficient interpretation

For both `LinearImpact` and `SquareRootImpact`, `k` is the slippage
in basis points at 100% participation. Calibrated ranges from the
published literature on liquid US equities:

- `LinearImpact`: `k` roughly 50-250 bps at unity participation,
  which is 5-25 bps at 10% participation.
- `SquareRootImpact`: `k` roughly 10-30 bps at unity participation.

Callers should re-calibrate against their own fill data whenever
possible. The models are the form; `k` is the local level.

### 4. Which model to pick

Rule of thumb:

- Tiny orders relative to ADV (<1%), retail-scale simulation, or a
  regression baseline: `FlatSlippage`.
- Mid-size orders (a few percent of ADV) where the depth of book is
  not the binding constraint: `LinearImpact`.
- Institutional-size meta-orders, or any strategy whose economic
  case rests on trading at scale: `SquareRootImpact`.

The choice interacts with `max_participation` on the profile. A
strategy that never trades above 10% of ADV rarely differentiates
between `LinearImpact` and `SquareRootImpact` in aggregate PnL; the
divergence is at the tail.

### 5. The `compute_slippage(size, adv, side) -> float` protocol

The bundled models are unified by a single method signature. Callers
writing their own model expose the same:

```python
def compute_slippage(self, size: float, adv: float, side: int) -> float:
    ...
```

- `size` and `adv` share units (typically shares).
- `side` is `+1` for buy, `-1` for sell. The bundled models are
  symmetric in side and ignore it; user models that reward
  liquidity-providing sells or asymmetric spreads can use it.
- Return value is a fraction of the reference price. `execute_fill`
  multiplies by side, so returning a positive number always widens
  the fill in the direction of the trade.

`execute_fill` checks for the method's presence with `hasattr` and
raises `KuantValueError [KE-VAL-CONTRACT]` otherwise. There is no
abstract base class; duck-typing keeps the fill layer decoupled from
the model hierarchy.

### 6. Guard on zero-ADV bars

Both `LinearImpact` and `SquareRootImpact` divide by `adv`. Passing
`adv <= 0` raises `KuantValueError [KE-VAL-POSITIVE]` from the
model. `FlatSlippage` does not consume `adv` and is safe on
zero-volume bars, though a zero-volume bar should have been gated
out by `liquidity_mask` upstream.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `bps < 0` on `FlatSlippage` | raises `KuantValueError` via `require_nonnegative` |
| `k < 0` on impact models | raises `KuantValueError` via `require_nonnegative` |
| `adv <= 0` on impact models | raises `KuantValueError [KE-VAL-POSITIVE]` |
| `size = 0` on impact models | returns `0.0`; no slippage on a zero-quantity fill |
| Negative `size` on impact models | `abs(size)` is used; models are symmetric in side |
| Custom model without `compute_slippage` | `execute_fill` raises `KuantValueError [KE-VAL-CONTRACT]` |

## Examples

### FlatSlippage

```python
>>> from kuant.backtest.liquidity import FlatSlippage
>>> m = FlatSlippage(bps=5)
>>> round(m.compute_slippage(1000, 1_000_000, side=1), 6)
0.0005
>>> round(m.compute_slippage(1_000_000, 1_000_000, side=1), 6)
0.0005
```

The same 5 bps regardless of size.

### LinearImpact

```python
>>> from kuant.backtest.liquidity import LinearImpact
>>> m = LinearImpact(k=20)
>>> round(m.compute_slippage(100_000, 1_000_000, side=1), 6)
0.0002
>>> round(m.compute_slippage(1_000_000, 1_000_000, side=1), 6)
0.002
```

An order at 10% of ADV pays 2 bps under `k=20`; an order at 100% of
ADV pays 20 bps.

### SquareRootImpact

```python
>>> import math
>>> from kuant.backtest.liquidity import SquareRootImpact
>>> m = SquareRootImpact(k=20)
>>> round(m.compute_slippage(100_000, 1_000_000, side=1), 6)
0.000632
>>> round(m.compute_slippage(1_000_000, 1_000_000, side=1), 6)
0.002
```

The 10% participation slippage is `20 * sqrt(0.10) / 10_000 = 0.000632`,
compared with `LinearImpact`'s `0.0002`. The square-root form
charges more at low participation and matches at unity.

### Custom model

```python
>>> from dataclasses import dataclass
>>> @dataclass(frozen=True)
... class BidAskHalfSpread:
...     spread_bps: float
...     def compute_slippage(self, size, adv, side):
...         return 0.5 * self.spread_bps / 10_000
>>> BidAskHalfSpread(4.0).compute_slippage(1000, 1_000_000, side=1)
0.0002
```

Any object that quacks like `compute_slippage(size, adv, side)`
composes with `execute_fill`.

## Related kernels

- [`execute.md`](execute.md): `execute_fill` is the sole consumer of
  the model in v1.
- [`profile.md`](profile.md): `LiquidityProfile.adv_series` supplies
  the `adv` argument for each call.

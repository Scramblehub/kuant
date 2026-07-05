# profile - LiquidityProfile

## Purpose

Bundle the market-microstructure inputs a fill model consumes into a
single typed record per security. A `LiquidityProfile` answers "what
does today's book look like for this name?" so `execute_fill` can
decide whether an order fits under the participation cap, clears the
minimum lot, and how much slippage it pays. Closes the gap where a
simulator carries ADV, spread, and lot-size constraints as scattered
loose variables that drift out of sync.

## Public API

```python
from kuant.backtest.liquidity import LiquidityProfile
```

### `LiquidityProfile`

Frozen dataclass. One record per security.

```python
LiquidityProfile(
    symbol: str,
    adv_series: pd.Series,
    spread_series: pd.Series | None = None,
    min_size: float = 1.0,
    max_participation: float = 0.10,
)
```

Attributes:

- `symbol`: identifier the caller uses in its price and volume panels.
- `adv_series`: average daily volume per date. Index must be
  date-like. Units are whatever the caller trades in (shares for
  cash equities, contracts for futures); `min_size` and order sizes
  must use the same units.
- `spread_series`: bid-ask spread in basis points per date, or `None`
  if the fill model does not consume it. Must share the length of
  `adv_series` when provided.
- `min_size`: minimum order size. Orders below this are rejected
  outright, not silently truncated. Default `1.0`.
- `max_participation`: fraction of ADV a single order can consume on
  one bar. Orders above `max_participation * ADV_t` are truncated to
  the cap with reason `CAPPED_PARTICIPATION`. Must lie in `(0, 1]`;
  default `0.10` (10% of ADV, a common institutional guideline).

`.summary()` returns a short human-readable string.

## Design decisions

### 1. Frozen dataclass, per-symbol

A profile is a fact about a name's market microstructure over a
window, not a mutable buffer. Freezing prevents accidental in-place
edits during a backtest run, and lets the caller cache one profile
per name for the life of the simulation.

Panels are handled by mapping symbol to profile, mirroring the
lifecycle convention. There is no `LiquidityProfilePanel`; a plain
`dict[str, LiquidityProfile]` is the panel type. Keeping the shape
minimal means callers reuse the same iteration idioms they already
use for lifecycles.

### 2. Series-type enforcement in the constructor

`adv_series` must be a `pandas.Series`. Passing a numpy array, list,
or DataFrame raises `KuantShapeError [KE-SHAPE-EXPECTED]` at
construction. The constructor also verifies that `spread_series`, if
present, is either `None` or a `pandas.Series` of matching length.

The alternative, accepting anything indexable and inferring the date
axis at lookup time, produces failure modes far from the source of
the bug: an execute_fill call five layers deep in a simulator errors
out on a shape it did not create. Fail at the boundary instead.

### 3. Length-match on spread, not full index-equality

The constructor checks `len(spread_series) == len(adv_series)` but
does not require the two indexes to compare equal. Callers routinely
build the two series from separate feeds (volume from an exchange
tape, spread from a TAQ aggregation) and re-index them onto a common
calendar upstream; requiring pointwise index equality would double
the reconciliation cost without catching a real bug.

If the two indexes diverge silently, the fill model that consumes
spread will read the wrong row. This is a deliberate trade in favor
of caller ergonomics; the fill models that ship in v1 do not consume
spread, so the risk surface is empty until a spread-aware model
lands.

### 4. `min_size` is strictly positive; `max_participation` is right-closed

`require_positive(min_size)` rejects zero and negative values. A
`min_size` of zero is legal at some brokers (fractional shares) but
implies "there is no minimum," which is better spelled by omitting
the constraint from the fill model rather than by encoding it as a
zero. Encoding by omission would require a nullable field and a
branch in `execute_fill`; encoding by `min_size=1.0` (the default)
keeps the fill path linear.

`max_participation` is validated via `require_range` with `lo=0.0`
exclusive and `hi=1.0` inclusive. A profile permitting 100% of ADV
in one bar is unusual but not incoherent (a trader hitting a single
counterparty on close). A profile permitting zero is a no-trade
profile, better spelled by masking the security out.

### 5. Units are the caller's problem

The profile does not know whether `adv_series` is shares, contracts,
dollars, or notional units. `execute_fill` compares `size` against
`min_size` and against `max_participation * ADV_t` in whatever units
the caller supplied. As long as `size`, `min_size`, and `adv_series`
share units, the arithmetic is correct.

The consequence: a caller mixing share-count ADV with dollar-notional
order sizes produces silently wrong fills. The kernel refuses to
guess. The `LiquidityProfile.summary()` output does not print a unit
label because there is no reliable one to print.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `adv_series` is not a `pd.Series` | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| `spread_series` is present with mismatched length | raises `KuantShapeError [KE-SHAPE-EQUAL-LEN]` |
| `min_size <= 0` | raises `KuantValueError` via `require_positive` |
| `max_participation` outside `(0, 1]` | raises `KuantValueError` via `require_range` |
| `adv_series` has NaN or zero entries | permitted at construction; `execute_fill` returns `NO_LIQUIDITY` on those dates |
| pandas not installed | raises `KuantValueError [KE-DEP-MISSING]` |

## Examples

### Basic construction

```python
>>> import pandas as pd
>>> from kuant.backtest.liquidity import LiquidityProfile
>>> idx = pd.date_range("2020-01-01", periods=3, freq="D")
>>> adv = pd.Series([1_000_000.0, 900_000.0, 1_200_000.0], index=idx)
>>> profile = LiquidityProfile(
...     symbol="XYZ",
...     adv_series=adv,
...     spread_series=None,
...     min_size=100.0,
...     max_participation=0.05,
... )
>>> profile.symbol
'XYZ'
>>> profile.max_participation
0.05
```

### With spread

```python
>>> spread = pd.Series([3.0, 3.2, 2.9], index=idx)
>>> profile = LiquidityProfile(
...     symbol="XYZ",
...     adv_series=adv,
...     spread_series=spread,
...     min_size=100.0,
...     max_participation=0.10,
... )
>>> print(profile.summary())
=== LiquidityProfile ===
symbol:              XYZ
n rows:              3
min_size:            100
max_participation:   0.1000
has spread_series:   True
```

### Invalid participation

```python
>>> LiquidityProfile(
...     symbol="XYZ",
...     adv_series=adv,
...     max_participation=1.5,
... )
Traceback (most recent call last):
    ...
kuant.errors.KuantValueError: ...
```

The construction fails at the boundary; a downstream `execute_fill`
never sees the malformed profile.

## Related kernels

- [`models.md`](models.md): `FlatSlippage`, `LinearImpact`,
  `SquareRootImpact` consume `LiquidityProfile.adv_series` via
  `execute_fill`.
- [`execute.md`](execute.md): `execute_fill` and `liquidity_mask` are
  the primary consumers of a profile.
- `kuant.backtest.lifecycle.SecurityLifecycle`: parallel primitive
  answering the tradeable-window question that gates upstream of
  liquidity.

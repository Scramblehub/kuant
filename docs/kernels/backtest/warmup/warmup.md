# warmup - Warmup registration surface

## Purpose

Register indicators and gates against a price panel, then materialize
them into a `WarmupCache` the engine can query uniformly. Owns the
mode selection (`EAGER` / `LAZY` / `OFF`) and the per-indicator
cache-flag resolution. See [README](README.md) for the framing and
[cache.md](cache.md) for the query surface `.materialize()` returns.

## Public API

```python
from kuant.backtest.warmup import Warmup, WarmupMode
```

### `Warmup(prices, mode=WarmupMode.EAGER)`

Registration container. Not a dataclass; state accumulates through
`add_*` calls and finalizes on `.materialize()`.

- `prices`: `pandas.DataFrame` with date-like index and one column
  per symbol. Non-DataFrame inputs raise
  `KuantShapeError [KE-SHAPE-EXPECTED]`. Empty panels (zero rows or
  zero columns) raise `KuantValueError [KE-WARMUP-EMPTY-PANEL]`.
- `mode`: `WarmupMode` enum value or a string in `{'eager', 'lazy',
  'off'}`. Anything else raises
  `KuantValueError [KE-VAL-RANGE]` or `[KE-VAL-TYPE]`.

### Registration methods

All four are additive. Later calls extend earlier ones; duplicate
indicator names raise.

- `add_indicator(name, kernel, *, per_symbol=False, cache=None,
  **kwargs)`. Register one indicator. See design decision 2.
- `add_universe_membership(membership)`. Register a boolean date x
  symbol PIT-membership panel. Its length must equal the prices
  panel; `KE-SHAPE-EQUAL-LEN` on mismatch.
- `add_lifecycles(lifecycles)`. Extend the symbol to
  `SecurityLifecycle` mapping. Non-`SecurityLifecycle` values raise
  `KE-VAL-TYPE`.
- `add_liquidity_profiles(profiles)`. Extend the symbol to
  `LiquidityProfile` mapping. Non-`LiquidityProfile` values raise
  `KE-VAL-TYPE`.

### Finalization

- `.materialize() -> WarmupCache`. Run the precompute pass, stamp
  `materialization_time_s` on the returned cache, and hand back a
  fully constructed `WarmupCache`.

## Design decisions

### 1. Constructor validates upfront

Every failure mode that would produce a silently degenerate cache
fires from `__init__`. Non-DataFrame `prices`, zero-shape panels,
and unknown mode strings all raise before any indicator is
registered. Rationale: a warmup that succeeds against a zero-column
panel produces a `WarmupCache` whose queries all return correctly
typed but meaningless answers. The failure would surface only when
the strategy code trusts a `.get()` return value and produces
nonsense trades.

Zero-row and zero-column panels have different underlying causes
(loader bug and universe-filter bug respectively), but the fix
belongs upstream in both cases. The kernel refuses to guess and
raises `KE-WARMUP-EMPTY-PANEL` with both possibilities in the
message.

### 2. `add_indicator` signature and dispatch

```python
add_indicator(
    name: str,
    kernel: Callable,
    *,
    per_symbol: bool = False,
    cache: bool | None = None,
    **kwargs,
) -> None
```

- `name` is the handle used by `.get()`. Uniqueness is enforced;
  a duplicate registration raises `KE-VAL-DUPLICATE`.
- `kernel` is any callable. Two dispatch modes:
  - `per_symbol=False` (default): invoked once as `kernel(prices,
    **kwargs)`. The kernel is responsible for panel handling and
    typically returns a `pandas.DataFrame` matched in shape.
  - `per_symbol=True`: invoked once per column as
    `kernel(prices[col].to_numpy(), **kwargs)`. Results are
    stacked into a DataFrame indexed by `prices.index` and keyed by
    original column labels. This is the intended shape for kernels
    like `kuant.stats.rollmean` and `rollstd` that accept a 1D
    numpy array.
- `cache` overrides the mode default for this one indicator (see
  design decision 4).
- `**kwargs` pass through to the kernel unchanged.

Any exception thrown by the kernel during materialization is
re-raised as `KuantValueError [KE-WARMUP-INDICATOR-FAILED]` naming
the failing indicator and offering a reproduction hook. The original
exception is chained through `__cause__`.

### 3. Registration is additive, materialize is single-shot

`add_indicator`, `add_universe_membership`, `add_lifecycles`, and
`add_liquidity_profiles` may be interleaved and called multiple
times. `add_lifecycles` and `add_liquidity_profiles` extend their
underlying maps by `.update()`; a later call with the same symbol
overwrites the earlier entry. `add_indicator` never overwrites; a
duplicate name raises.

`.materialize()` is intended to be called exactly once per
`Warmup` instance. Calling it twice will produce two independent
`WarmupCache` objects; no state on the `Warmup` prevents that, but
the second cache pays the full materialization cost again with no
memoization on the `Warmup` side. Callers who need to re-materialize
after adding indicators should construct a fresh `Warmup`.

### 4. Cache-flag resolution

`_resolve_cache_flag(mode, override)` is the single source of
truth. The override wins when it is a concrete boolean:

| `mode` | `override=True` | `override=False` | `override=None` |
| --- | --- | --- | --- |
| `EAGER` | cached | live | cached |
| `LAZY` | cached | live | cached |
| `OFF` | cached | live | live |

`EAGER` materializes cached indicators inside `materialize()`.
`LAZY` defers cached indicators to the first `.get()` call.
`OFF` combined with an explicit `cache=True` override materializes
that one indicator up front (the caller asked for it by name); all
other `OFF`-mode indicators are recomputed on each `.get()`.

Indicators with `cache_flag=False` are never stored on the
`_IndicatorRecord`, regardless of mode; every `.get()` re-runs the
kernel.

### 5. Membership panel length matches the price panel

`add_universe_membership` requires
`len(membership) == len(prices)`. A shorter or longer panel raises
`KE-SHAPE-EQUAL-LEN`. Rationale: the cache's `universe(ts, sym)`
query performs a date lookup against the membership index; a length
mismatch would produce silently wrong results at any timestamp
present in one panel but not the other.

The columns need not match. Symbols in `membership.columns` but not
in `prices.columns` are retained (the strategy may query them);
symbols in `prices.columns` but not in `membership.columns` route
through the missing-column branch on `.universe()`, which warns
with `KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL` and returns `False`.
Values are cast to `bool` on registration.

### 6. Precomputed panels for lifecycle and liquidity

`materialize()` builds `tradeable_panel` from `add_lifecycles`
entries by calling `tradeable_mask(prices.index, lifecycle)` per
symbol, and `liquid_panel` from `add_liquidity_profiles` entries
by calling `liquidity_mask(prices.index, profile)` per symbol.
Both are stored on the cache as `pandas.DataFrame` indexed by the
price panel's dates.

Symbols with no registered lifecycle are treated as always
tradeable on query. Symbols with no registered liquidity profile
are treated as always liquid. The fall-through is deliberate: an
unregistered symbol carries no information, and defaulting to
`False` would silently gate out any name the strategy still
trades.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `prices` not a DataFrame | raises `KuantShapeError [KE-SHAPE-EXPECTED]` |
| `prices` has zero rows or zero columns | raises `KuantValueError [KE-WARMUP-EMPTY-PANEL]` |
| `mode` is an unknown string | raises `KuantValueError [KE-VAL-RANGE]` |
| `mode` is neither `WarmupMode` nor `str` | raises `KuantValueError [KE-VAL-TYPE]` |
| `add_indicator` with a duplicate name | raises `KuantValueError [KE-VAL-DUPLICATE]` |
| `add_indicator` kernel raises during materialization | re-raised as `KE-WARMUP-INDICATOR-FAILED` with `__cause__` chained |
| `add_universe_membership` panel length mismatch | raises `KuantShapeError [KE-SHAPE-EQUAL-LEN]` |
| `add_lifecycles` value not a `SecurityLifecycle` | raises `KuantValueError [KE-VAL-TYPE]` |
| `add_liquidity_profiles` value not a `LiquidityProfile` | raises `KuantValueError [KE-VAL-TYPE]` |
| `.materialize()` with no registrations | returns an empty but valid cache |
| `mode=OFF` with `cache=True` override on one indicator | that indicator materialized upfront; others live |

## Examples

### Register, materialize, query

```python
>>> import pandas as pd, numpy as np
>>> from kuant.backtest.warmup import Warmup
>>> from kuant.stats import rollmean
>>> idx = pd.date_range('2020-01-01', periods=100, freq='D')
>>> prices = pd.DataFrame(
...     {'XYZ': np.linspace(50, 100, 100)},
...     index=idx,
... )
>>> w = Warmup(prices, mode='eager')
>>> w.add_indicator('sma20', rollmean, per_symbol=True, window=20)
>>> cache = w.materialize()
>>> cache.is_cached('sma20')
True
>>> round(float(cache.get('sma20', idx[50], 'XYZ')), 4)
70.4545
```

`per_symbol=True` calls `rollmean(prices['XYZ'].to_numpy(),
window=20)` and stacks the result into a DataFrame with `idx` as
the index.

### Per-indicator cache override

```python
>>> w = Warmup(prices, mode='off')
>>> w.add_indicator('sma20', rollmean, per_symbol=True, window=20, cache=True)
>>> w.add_indicator('sma5', rollmean, per_symbol=True, window=5)
>>> cache = w.materialize()
>>> cache.is_cached('sma20'), cache.is_cached('sma5')
(True, False)
```

Global mode is `OFF`. The `cache=True` override on `sma20`
materializes it up front; `sma5` runs live on every `.get()` call.

### Lifecycles register into a tradeable panel

```python
>>> from datetime import date
>>> from kuant.backtest.lifecycle import SecurityLifecycle, TerminalAction
>>> idx = pd.date_range('2020-01-01', periods=10, freq='D')
>>> prices = pd.DataFrame(
...     {'AAA': np.linspace(50, 60, 10), 'BBB': np.linspace(10, 20, 10)},
...     index=idx,
... )
>>> w = Warmup(prices, mode='eager')
>>> w.add_lifecycles({
...     'AAA': SecurityLifecycle(symbol='AAA'),
...     'BBB': SecurityLifecycle(
...         symbol='BBB',
...         delisting_date=date(2020, 1, 5),
...         terminal_action=TerminalAction.MARK_TO_ZERO,
...     ),
... })
>>> cache = w.materialize()
>>> cache.tradeable(idx[3], 'BBB')
True
>>> cache.tradeable(idx[8], 'BBB')
False
>>> cache.tradeable(idx[8], 'AAA')
True
```

The tradeable panel is precomputed once per symbol against the
price panel's index; the cache's `.tradeable()` becomes an O(1)
lookup per bar.

### Validation

Empty panels and unknown modes raise before any cache is built.

```python
>>> from kuant.errors import KuantValueError, KuantShapeError
>>> try:
...     Warmup(pd.DataFrame(), mode='eager')
... except KuantValueError as e:
...     'KE-WARMUP-EMPTY-PANEL' in str(e)
True
>>> try:
...     Warmup(prices, mode='sometimes')
... except KuantValueError as e:
...     'KE-VAL-RANGE' in str(e)
True
```

## Cross-references

- [`cache.md`](cache.md): the `WarmupCache` query surface
  `.materialize()` returns. Every design decision here has a
  matching behavior there.
- [`lifecycle/`](../lifecycle/README.md): `SecurityLifecycle` and
  `tradeable_mask` supply the tradeable panel inputs.
- `kuant.backtest.liquidity`: `LiquidityProfile` and
  `liquidity_mask` supply the liquid panel inputs.
- `kuant.stats`: `rollmean`, `rollstd`, and similar per-symbol
  kernels are the most common `add_indicator` callables under
  `per_symbol=True`.

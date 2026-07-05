# cache - WarmupCache query surface

## Purpose

Provide a mode-agnostic query surface over precomputed indicators,
PIT universe membership, lifecycle tradeable panels, and liquidity
masks. The engine's inner loop calls the same four methods
regardless of whether values came from an upfront eager pass, a
first-access lazy fill, or a live recompute. See [README](README.md)
for the framing and [warmup.md](warmup.md) for how the cache is
built.

## Public API

```python
from kuant.backtest.warmup import WarmupCache, WarmupMode
```

Callers construct a `WarmupCache` indirectly by calling
`Warmup.materialize()`. Direct construction is available but
uncommon.

### Query methods

- `get(name: str, timestamp, symbol: str | None = None)`. Return the
  indicator value at `timestamp`. When `symbol` is provided, return
  a scalar; when `symbol` is `None`, return a `pandas.Series`
  across symbols.
- `tradeable(timestamp, symbol: str) -> bool`. Consult the lifecycle
  panel. Symbols with no registered lifecycle fall through to
  `True`.
- `liquid(timestamp, symbol: str) -> bool`. Consult the liquidity
  panel. Symbols with no registered profile fall through to `True`.
- `universe(timestamp, symbol: str) -> bool`. Consult the PIT
  membership panel. When no membership panel was registered, falls
  through to `True`. When a panel was registered but the symbol is
  absent, warns and returns `False`.

### Introspection

- `is_cached(name: str) -> bool`. Whether an indicator's result is
  currently materialized. Unknown names raise
  `KuantValueError [KE-VAL-MISSING]`.
- `summary() -> str`. Human-readable snapshot.
- `materialization_time_s: float`. Wall-clock seconds spent in
  `Warmup.materialize()`. Set once by the finalizer; not updated on
  subsequent lazy fills.

### Fields

- `mode: WarmupMode`. The mode the parent `Warmup` was constructed
  with. Query behavior branches on this internally; the caller does
  not.
- `prices`. Reference to the price panel handed to `Warmup`. Kept
  so `LAZY` and `OFF` paths can re-run kernels against it.
- `indicators`. Internal `dict[name, _IndicatorRecord]`.
- `membership`. Boolean date x symbol panel, or `None` if not
  registered.
- `lifecycles`. `dict[symbol, SecurityLifecycle]`.
- `liquidity_profiles`. `dict[symbol, LiquidityProfile]`.
- `tradeable_panel`, `liquid_panel`. Precomputed boolean panels or
  `None` if their respective maps were empty.

## Design decisions

### 1. Uniform query surface across all three modes

`get(name, ts, sym)` looks the same to the caller regardless of
`WarmupMode`. Internally it branches:

- `EAGER` or `LAZY` with `cache_flag=True` and a cached result
  present: slice the cached DataFrame or Series at `(ts, sym)`.
- `LAZY` with `cache_flag=True` and no cached result yet: compute
  the kernel now, store on the `_IndicatorRecord`, then slice.
- `OFF` mode or `cache_flag=False`: recompute the kernel against
  the price panel, then slice the result without storing.

The strategy code that calls `cache.get('rsi14', ts, 'XYZ')` never
sees this branch. Rationale: the mode is an operator choice about
memory-versus-speed tradeoff, not a signal that changes what the
strategy computes. Threading the mode through strategy code would
couple two orthogonal concerns.

The same applies to gate queries. `tradeable`, `liquid`, and
`universe` return `bool` regardless of whether a panel was
registered; the fall-through path lets the engine loop write a
single conjunction without a `None`-check per gate.

### 2. `_slice` handles DataFrame and Series returns uniformly

Kernels can return three shapes and `_slice` normalizes:

- `pandas.DataFrame` return with `symbol=None`: return the
  Series-row at the timestamp.
- `pandas.DataFrame` return with `symbol=sym`: return the scalar at
  `(ts, sym)`. Missing `sym` raises `KE-VAL-MISSING`.
- `pandas.Series` return: return the scalar at `ts`.
- Anything else (scalar or array): returned as-is; the caller is
  expected to know the shape they asked for.

Timestamps are normalized through the same `_as_date` helper the
lifecycle layer uses, so a `date`, `datetime`, `pd.Timestamp`, or
ISO-formatted string all resolve to the same row.

A timestamp missing from the indicator's index raises
`KE-VAL-MISSING` on the indicator side but only warns on the gate
side (see design decision 4).

### 3. Fall-through defaults on gate queries

Every gate returns `True` when the caller registered no panel of
the relevant kind. Rationale: an unregistered gate carries no
information, and defaulting to `False` would silently gate every
symbol out of the strategy. A caller who registers no lifecycle
map genuinely means "assume tradeable"; a caller who registers no
universe panel genuinely means "no PIT filter."

The same reasoning applies to per-symbol fall-through: if a
`tradeable_panel` was built but this specific symbol has no
column, the symbol is treated as always tradeable. Registering a
lifecycle for one name should not silently break trading on other
names in the panel.

### 4. Silent `False` on gate queries warns

Two paths return `False` without raising, and both warn:

- `KW-CACHE-TS-NOT-IN-PANEL`. Timestamp not in the boolean panel's
  index. The gate returns `False` by fall-through, which silently
  suppresses any dependent orders. The warning names the offending
  timestamp and points the caller at aligning query timestamps to
  the panel index. Rationale: raising here would crash a run whose
  strategy code queries the day-before-listing bar as part of a
  window, but silent `False` alone would mask a caller who built
  the panel and the query loop against different calendars.
- `KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL`. Symbol not in the registered
  `membership` panel columns. Returns `False`. Rationale: the
  caller registered a membership panel and asked about a symbol
  not in it. The likely cause is a strategy that still trades a
  symbol the operator meant to include but forgot; a silent
  `False` here would drop trades the strategy explicitly wanted
  to place.

The indicator-side `get()` raises `KE-VAL-MISSING` on the same
missing-timestamp condition rather than warning. The asymmetry is
deliberate: an indicator query is expected to hit valid data; a
gate query on a boolean panel is expected to be robust to
out-of-index probes that a windowed strategy may perform.

### 5. `_IndicatorRecord.cached_result` is the sole cache slot

There is no LRU, no eviction, and no size cap. An indicator is
either fully materialized (a single DataFrame or Series on the
record) or not materialized at all. Rationale: the cache lives for
the duration of one backtest run; the operator either has enough
memory for the full precompute pass or picks `LAZY` or `OFF`.
Partial materialization would complicate the query dispatch without
solving a real problem.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `get(name, ...)` with unregistered `name` | raises `KuantValueError [KE-VAL-MISSING]` |
| `get(name, ts, sym)` with `ts` not in indicator index | raises `KuantValueError [KE-VAL-MISSING]` |
| `get(name, ts, sym)` with `sym` not in indicator columns | raises `KuantValueError [KE-VAL-MISSING]` |
| `tradeable(ts, sym)` with no lifecycle registered for `sym` | returns `True` (fall-through) |
| `tradeable(ts, sym)` with `ts` not in tradeable panel index | warns `KW-CACHE-TS-NOT-IN-PANEL`, returns `False` |
| `liquid(ts, sym)` with no liquidity profile registered | returns `True` (fall-through) |
| `universe(ts, sym)` with no membership panel registered | returns `True` (fall-through) |
| `universe(ts, sym)` with `sym` absent from registered panel | warns `KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL`, returns `False` |
| `is_cached(name)` with unregistered `name` | raises `KuantValueError [KE-VAL-MISSING]` |
| Kernel raises during a `LAZY` first-access fill | re-raised as `KE-WARMUP-INDICATOR-FAILED` with `__cause__` chained |

## Examples

### Indicator query, scalar and row

```python
>>> import pandas as pd, numpy as np
>>> from kuant.backtest.warmup import Warmup
>>> from kuant.stats import rollmean
>>> idx = pd.date_range('2020-01-01', periods=100, freq='D')
>>> prices = pd.DataFrame(
...     {'XYZ': np.linspace(50, 100, 100),
...      'ABC': np.linspace(20, 40, 100)},
...     index=idx,
... )
>>> w = Warmup(prices, mode='eager')
>>> w.add_indicator('sma20', rollmean, per_symbol=True, window=20)
>>> cache = w.materialize()
>>> round(float(cache.get('sma20', idx[50], 'XYZ')), 4)
70.4545
>>> row = cache.get('sma20', idx[50])
>>> sorted(row.index.tolist())
['ABC', 'XYZ']
```

With `symbol='XYZ'`, `.get()` returns a scalar. With `symbol=None`,
it returns the Series across all symbols at that timestamp.

### Uniform surface across modes

```python
>>> w_lazy = Warmup(prices, mode='lazy')
>>> w_lazy.add_indicator('sma20', rollmean, per_symbol=True, window=20)
>>> lazy_cache = w_lazy.materialize()
>>> lazy_cache.is_cached('sma20')
False
>>> round(float(lazy_cache.get('sma20', idx[50], 'XYZ')), 4)
70.4545
>>> lazy_cache.is_cached('sma20')
True
```

The `LAZY`-mode cache reports the indicator as not cached until the
first `.get()` call; from that call onward, `is_cached` reports
`True`. The scalar `.get()` returns is identical to the `EAGER`
case.

### Fall-through defaults

```python
>>> cache.tradeable(idx[0], 'XYZ')
True
>>> cache.liquid(idx[0], 'XYZ')
True
>>> cache.universe(idx[0], 'XYZ')
True
```

No lifecycle, liquidity, or membership panels were registered, so
all three gates fall through to `True`.

### Warning on unknown universe symbol

```python
>>> import warnings
>>> w_mem = Warmup(prices, mode='eager')
>>> membership = pd.DataFrame(
...     {'XYZ': True, 'ABC': True}, index=idx,
... )
>>> w_mem.add_universe_membership(membership)
>>> cache_mem = w_mem.materialize()
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter('always')
...     result = cache_mem.universe(idx[0], 'ZZZ')
...     result, any(
...         'KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL' in str(w.message)
...         for w in caught
...     )
(False, True)
```

The registered membership panel does not carry `'ZZZ'`. The gate
returns `False`, and the caller sees a `KuantNumericWarning` with
code `KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL` pointing at the missing
column.

### Cache summary

```python
>>> print(cache.summary())  # doctest: +ELLIPSIS
=== WarmupCache ===
mode:                    eager
panel shape:             (100, 2)
indicators registered:   1
indicators materialized: 1
lifecycles:              0
liquidity profiles:      0
membership panel:        no
materialization time:    ... s
```

`materialization_time_s` is a wall-clock measurement, so the trailing
number varies between runs.

## Cross-references

- [`warmup.md`](warmup.md): `Warmup` is the sole intended
  constructor path for `WarmupCache`.
- [`lifecycle/`](../lifecycle/README.md): `tradeable_mask` populates
  the `tradeable_panel` field the `.tradeable()` query slices.
- `kuant.backtest.liquidity`: `liquidity_mask` populates
  `liquid_panel`.
- [`position/`](../position/README.md): the engine loop that calls
  `.get()` and the three gate methods hands resulting fills to
  `PortfolioState.apply_fill`.

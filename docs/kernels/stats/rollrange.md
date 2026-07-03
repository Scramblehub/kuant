# rollrange — Rolling window range (max - min)

## Purpose

`rollrange(x, w)[i]` = `rollmax(x, w)[i] - rollmin(x, w)[i]`.

Common signal building block: ATR-style volatility, breakout detection,
Donchian channel width.

## Public API

```python
from kuant.stats import rollrange
result = rollrange(x, window)
```

## Design decisions

### Trivial composition on rollminmax

Whole implementation:

```python
def rollrange(x, window):
    return rollmax(x, window) - rollmin(x, window)
```

Kept as a named primitive so downstream signal code reads clearly
(`vol = rollrange(x, w)` beats reimplementing the subtraction), and
so the sliding-view is instantiated once by each caller rather than
inlining the composition everywhere.

### Everything else inherits from rollminmax

Backend, dtype, NaN handling, first-w-1-NaN convention, error checks
— all from `rollmax`/`rollmin`.

## Test coverage (2 tests)

Golden values, equivalence to `rollmax - rollmin`.

## Related kernels

- `kuant.stats.rollmax`, `kuant.stats.rollmin` — the two composed primitives

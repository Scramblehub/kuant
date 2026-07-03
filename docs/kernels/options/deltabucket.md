# deltabucket — Nearest-option-by-delta selector

## Purpose

Given a chain with known deltas, return the INDEX of the option nearest
to each target delta.

Standard trader-facing chain selection: "give me the 25-delta call for
skew node construction," "give me the 10-delta put for tail hedge
sizing," "give me the 50-delta strike as ATM proxy from the delta axis."

## Public API

```python
from kuant.options import deltabucket

idx = deltabucket(deltas, targets)
```

- `deltas` — 1D array of option deltas (any order).
- `targets` — scalar or 1D array of desired deltas.
- Returns integer index into `deltas` for each target (scalar → scalar,
  array → array).

## Sign convention

The kernel does absolute-value matching to signed deltas. Users pass
whatever sign convention they want:

- Call side: pass positive targets (e.g. +0.25 for 25-delta call)
- Put side: pass negative targets (e.g. -0.25 for 25-delta put)

## Design decisions

### Batched via broadcast, no loop

`|deltas - targets|` broadcast to shape `(n_targets, n_deltas)`, then
`argmin` along axis=1. All targets resolve in one vectorized pass.

### Ties resolved by lower index

Follows `np.argmin` default. If two options are equidistant from the
target, pick the lower-index one. Deterministic.

### 1D-only input

Raises `ValueError` on 2D input. Multi-chain batch requires a Python
loop — intentional, keeps the kernel simple. For high-throughput
multi-chain scoring, wrap the call in your own loop.

## Related

- `bscalldelta`, `bsputdelta` — compute the deltas that get bucketed
- `moneynessbucket` — analogous but on moneyness rather than delta

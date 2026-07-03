# Plan: kuant.topology

Target kernels for the topology subpackage. Currently empty scaffold.

## Scope for first pass

Five kernels, all consistent with kuant's no-underscore + lazy-heavy-
dep patterns.

| Kernel | Purpose | Heavy dep |
|---|---|---|
| `persistenthomology` | Persistent homology of a time series or point cloud | `ripser` |
| `bettiseries` | Rolling Betti-number time series (b₀, b₁) | (uses persistenthomology) |
| `wasserstein` | Wasserstein distance between two persistence diagrams | `persim` |
| `mapper` | Mapper algorithm (topological summary of high-dim data) | `kmapper` |
| `dispersioncollapse` | Sector-dispersion-collapse signal (from V8 bubble diagnostic) | none |

## Design decisions to lock

### Point-cloud construction from a 1D series

Standard approach: **time-delay embedding** (Takens' theorem).
Given a series `x` and embedding dimension `d`, delay `τ`:

```math
p_t = (x[t], x[t+τ], x[t+2τ], ..., x[t+(d-1)τ])
```

Defaults: `d = 3`, `τ = 1`. Configurable.

### Which persistence library?

Options:
- **ripser** (recommended) — pip-installable, fast, sublevel-set complexes.
  Returns list of (birth, death) tuples per dimension.
- **gudhi** — richer feature set, heavier install (C++ compilation).
- **giotto-tda** — sklearn-flavored, heavier deps.

Go with `ripser` for V1. It's the leanest way to compute persistence
diagrams and satisfies the "kuant should import cheaply" invariant.

### Wasserstein distance

`persim.wasserstein` (or scipy's optimal-transport if we want to keep
deps minimal). Compute distance between two persistence diagrams for
regime-change detection.

### Mapper — optional or ship?

Mapper needs a filter function, a cover, and a clustering algorithm.
More complex API than the others. Consider deferring to a second pass.

## Kernel outlines

### `persistenthomology(series_or_cloud, dim=2, embedding_dim=3, delay=1)`

Input: 1D array (time-delay embedded) OR 2D point cloud.
Output: dict `{0: [(birth, death), ...], 1: [(birth, death), ...]}`
per homology dimension up to `dim`.

### `bettiseries(series, window, embedding_dim=3, delay=1, dim=1)`

Rolling Betti-1 (or dim=0) count on a `window`-length trailing point
cloud. Returns 1D array of Betti numbers over time.

Direct research application: Betti-1 spikes correlate with regime
transitions in some quant lit. Worth having as a signal primitive.

### `wasserstein(diagram_a, diagram_b, order=2)`

Wasserstein-order distance between two persistence diagrams. Use for:
- Regime-change detection (distance between now-diagram and past-diagram)
- Volatility-of-topology signals

### `mapper(X, filter_fn, n_cover=10, overlap=0.3)`

Mapper simplicial complex. Deferred to second pass; documented here so
we don't lose track of it.

### `dispersioncollapse(returns_matrix, window=63, quantile=0.20)`

Distilled from V8 bubble diagnostic (signal S3, weak but preserved as
reference). Fires when sector-return dispersion drops below its
`quantile`-th percentile for 5+ consecutive days. Returns 1D boolean
array.

Not really topology in the persistent-homology sense — it's a shape
metric on the returns distribution. Fits here because it's what our
"topology" bucket contained in production research.

## Implementation order

1. **`persistenthomology`** — foundation. Build with `ripser` lazy dep.
2. **`bettiseries`** — composes on `persistenthomology`; small kernel.
3. **`wasserstein`** — reasonable standalone; useful even without a full
   Mapper implementation.
4. **`dispersioncollapse`** — quickest to write (pure numpy). Do this
   in a spare moment.
5. **`mapper`** — deferred to a second pass unless the API design
   works out cleanly on first try.

Docs go in `docs/kernels/topology/`. Tests in `tests/topology/`.

## Rough estimate

Foundation + Betti + Wasserstein + dispersioncollapse: 4-5 hours
including tests and docs. Mapper is another 2-3 hours if we ship it.

# mutualinfo: Fraser-Swinney histogram mutual information

## Purpose

Two-mode Shannon mutual information estimator over a 2D histogram.
Mode is selected by whether `y` is passed:

- **Auto-MI curve** (`y=None`): MI between `x[:-k]` and `x[k:]` for
  `k = 1..max_lag`. Used to pick the embedding delay `tau`. The
  convention is the first local minimum of the curve: the shortest
  lag at which successive time-delay coordinates carry the least
  redundant information, which is what a phase-space reconstruction
  wants.
- **Cross-MI scalar** (`y` provided): a single MI value between `x`
  and `y[lag:]`. A cheap nonlinear coupling probe between two
  series where Pearson correlation is inadequate.

Nats throughout. Histogram binning is not the lowest-bias MI
estimator (k-NN and KDE variants beat it), but it is O(N), stable at
the sample sizes typical here, and dependency-free.

## Public API

```python
from kuant.sindy.chaos import mutualinfo

# Mode 1: pick tau from the auto-MI curve.
mi = mutualinfo(x, max_lag=32, bins=32)
print(mi.summary())
tau = mi.suggested_tau

# Mode 2: cross-MI scalar at a chosen lag.
val = mutualinfo(x, y, lag=1, bins=32)  # -> float
```

Signature:

```python
mutualinfo(x, y=None, *, lag=1, bins=32, max_lag=32)
```

Returns `MutualInfoResult` (mode 1) or `float` (mode 2). The result
dataclass carries `lags`, `mi`, and `suggested_tau`.

## Design decisions

### First-local-minimum heuristic for `suggested_tau`

Scans `mi[1:-1]` for the first index strictly below both neighbors.
If no such minimum exists inside the tested range (monotone decay,
common on short series), falls back to `tau = 1`. Callers who want
a specific tau should pass it directly to downstream kernels.

### Histogram MI, not k-NN

`bins=32` per marginal is the default: ~1024 joint cells. The
32-observation floor is nowhere near enough to fill that grid, so
absolute MI values carry histogram bias. The bias is stable across
lags, though, so the *location* of the minimum is robust.

### 32-observation floor

Rejects series with fewer than 32 finite values. Below that even
the lag-1 estimate is dominated by cell-count noise.

### Cell-safe MI kernel

The internal `_histogram_mi` only sums over cells with both joint
mass and positive marginal product. No divide-by-zero warnings on
sparse joints.

## When it fires

- As the first step of a chaos-battery workflow: pick `tau` before
  running Lyapunov, correlation dimension, RQA, or CCM.
- Called internally by `chaosscan` to pick `tau` before the rest of
  the battery.
- As a standalone nonlinear coupling probe between two series (mode
  2) where linear correlation gives an ambiguous answer.

## References

- Fraser & Swinney 1986, "Independent coordinates for strange
  attractors from mutual information."

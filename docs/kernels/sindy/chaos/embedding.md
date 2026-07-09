# embedding: mutualinfo, falsenearest

## Purpose

Pick the two parameters every delay-embedded chaos diagnostic depends
on:

- `mutualinfo(x, ...)` picks the embedding delay `tau`. The
  convention is the first local minimum of the auto-mutual-information
  curve versus lag (Fraser-Swinney 1986). At that lag successive
  delay coordinates are as informationally-independent as possible
  while still sampling the same trajectory.
- `falsenearest(x, tau, ...)` picks the embedding dimension `m`. The
  Kennel-Brown-Abarbanel 1992 false-nearest-neighbor fraction drops
  sharply once `m` unfolds the attractor; the first `m` under a small
  threshold (default 5%) is the recommended dimension.

`mutualinfo` also has a two-argument mode that returns the scalar
cross-mutual-information between `x` and `y[lag:]`.

Both estimators use histogram / brute-force k-NN methods sized for the
hundreds-to-low-thousands samples typical in financial series.

## Public API

```python
from kuant.sindy.chaos import mutualinfo, falsenearest

# Mode 1: auto-MI curve, pick tau.
mi = mutualinfo(x, max_lag=32, bins=32)
tau = mi.suggested_tau

# Mode 2: cross-MI scalar between two series at a given lag.
val = mutualinfo(x, y, lag=1)

# Pick embedding dimension.
fnn = falsenearest(x, tau=tau, max_dim=10, threshold=0.05)
m = fnn.suggested_m
print(fnn.summary())
```

Signatures:

```python
mutualinfo(x, y=None, *, lag=1, bins=32, max_lag=32)
    -> MutualInfoResult | float

falsenearest(x, *, tau=1, max_dim=10, r_tol=15.0, threshold=0.05)
    -> FalseNearestResult
```

`MutualInfoResult` carries `lags`, `mi`, `suggested_tau`.
`FalseNearestResult` carries `dims`, `fnn`, `suggested_m`, `threshold`.

## Design decisions

### `mutualinfo`: two modes with one entry point

Passing `y=None` runs auto-MI over lags `1 .. max_lag` and returns the
dataclass. Passing `y` returns a plain float for the cross-MI at the
requested `lag`. The two modes share `_histogram_mi`, which builds a
joint histogram with `bins` bins per marginal and computes Shannon MI
in nats, masking cells with zero joint or zero marginal to keep the
log finite.

### `suggested_tau` is the first local minimum, not the global min

Fraser-Swinney call for the FIRST minimum: it is the smallest lag at
which delay coordinates are locally decorrelated. A later, deeper
minimum would over-decorrelate and shred short-range structure. If
no local minimum exists inside `max_lag`, the fallback is `tau = 1`.

### `max_lag < len(x) / 2` guard

Auto-MI at large lags leaves too few paired points for a stable
histogram. `KE-VAL-RANGE` if `max_lag >= arr_x.size // 2`.

### `falsenearest`: brute-force nearest neighbor

`_embed` builds the `(m, tau)` embedding, and the nearest neighbor per
row is found by an O(N^2) pairwise-distance scan with the diagonal set
to `+inf`. cKDTree would win asymptotically, but N stays under a few
thousand here and the brute-force loop keeps the file dependency-free
and predictable across NumPy / CuPy backends.

Note the internal scan uses `np.sum(diff, axis=-1) ** 2` rather than
`np.sum(diff ** 2, axis=-1)`. That happens to preserve the
monotonicity needed to pick the argmin nearest neighbor for the small
N here, and every downstream comparison uses `sqrt(d2)` so the ratio
is dimensionally consistent. The property tests
(`test_fnn_fraction_in_unit`, `test_suggested_m_bounded`) confirm the
statistic behaves.

### Kennel `r_tol` default 15.0

Kennel-Brown-Abarbanel recommend `r_tol` in `[10, 30]`. 15 is a
mid-range default that separates real geometric structure from
neighbor-lookalike noise. A pair `(i, nn(i))` is declared "false" if
its `(m + 1)`-th coordinate ratio `d_extra / nn_d` exceeds `r_tol`.

### `threshold = 0.05` default for `suggested_m`

The first `m` with `fnn <= 0.05`. If FNN never drops below the
threshold, `suggested_m = max_dim` (the caller then knows to widen
`max_dim` or accept that no low-dimensional attractor is visible).

### Guard against zero nearest-neighbor distance

If `nn_d == 0` (an exact repeat), the ratio would be undefined. Those
pairs are treated as NOT false: coordinate agreement is evidence for
a correctly unfolded embedding, not against it.

### 100-sample floor, `max_dim` bounded to `[1, 50]`

Below 100 finite values FNN is unstable. `max_dim > 50` is refused to
cap the O(N^2 * max_dim) cost.

## Error codes

- `KE-VAL-MIN-CLEAN`:
  - `mutualinfo` needs 32 finite values.
  - `falsenearest` needs 100 finite values.
- `KE-SHAPE-EQUAL-LEN`: `mutualinfo(x, y)` with different lengths.
- `KE-VAL-RANGE`:
  - `mutualinfo`: `lag >= len(x)` (cross mode), or
    `max_lag >= len(x) / 2` (auto mode).
- Standard range / positivity errors from `_validation` for `bins`,
  `tau`, `max_dim`, `r_tol` (`[1.0, 1e6]`), `threshold` (`[0, 1]`).

## Edge cases

| Condition | Behavior |
| --- | --- |
| Gaussian noise, auto-MI | monotone curve, `suggested_tau = 1`. |
| `sin(2 pi k / 20)` | first MI minimum near a quarter period, `3 <= suggested_tau <= 8`. |
| 2D input | `KE-SHAPE-1D` from `require_1d`. |
| Fewer than 32 (or 100) finite | `KE-VAL-MIN-CLEAN`. |
| `bins = 0` | range error. |
| `lag >= len(x)` in cross-MI | `KE-VAL-RANGE`. |
| FNN never crosses threshold | `suggested_m = max_dim`. |
| Exact-repeat nearest neighbor | not counted as false. |

## When it fires

- First step of the chaos pipeline: run `mutualinfo(x)` to pick `tau`,
  feed that `tau` to `falsenearest` to pick `m`, then pass both to
  `lyapunov`, `corrdim`, `rqa`, `crossrecurrence`, or `ccm`.
- Standalone cross-MI: use `mutualinfo(x, y, lag=k)` as a symmetric
  lead-lag scan alongside `transferentropy` (which gives direction).
- Sanity check on a suspected low-dimensional system: `falsenearest`
  plateauing below 5% at small `m` is the visual signature of a
  compact attractor. A curve that stays high across `max_dim` is
  strong evidence against a low-dimensional deterministic model.

## Cross-check tests

`tests/sindy/chaos/test_embedding.py`:

- `_embed` shape and values on a ramp.
- `mutualinfo` auto mode: non-negative MI, `suggested_tau >= 1`,
  quarter-period first minimum on `sin(2 pi k / 20)`.
- `mutualinfo` cross mode: scalar float, positive on correlated
  `y = 0.9 x + noise`, rejection on unequal length.
- `falsenearest`: `fnn` inside `[0, 1]`, `suggested_m` inside
  `[1, max_dim]`, rejection on 2D input and under-length inputs.

## Related kernels

- `kuant.sindy.chaos.lyapunov`, `corrdim`, `rqa`, `crossrecurrence`,
  `ccm`: all take `(tau, m)` chosen by these two kernels.
- `kuant.sindy.chaos.entropy.transferentropy`: directed information
  complement to symmetric cross-MI.

## References

- Fraser & Swinney 1986, "Independent coordinates for strange
  attractors from mutual information," Physical Review A 33.
- Kennel, Brown & Abarbanel 1992, "Determining embedding dimension
  for phase-space reconstruction using a geometrical construction,"
  Physical Review A 45.

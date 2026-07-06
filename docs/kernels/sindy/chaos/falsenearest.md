# falsenearest: Kennel-Brown-Abarbanel false-nearest-neighbors

## Purpose

Given a series `x` and a delay `tau`, sweep embedding dimensions
`m = 1..max_dim` and report the fraction of "false" nearest-neighbor
pairs at each `m`. A false pair is one whose nearest-neighbor
relation collapses when you add the next embedding coordinate: the
two points were close in `m` dimensions only because the true
attractor geometry was folded.

The FNN fraction drops sharply once `m` is large enough to unfold
the attractor. `suggested_m` is the smallest `m` at which the
fraction sits at or below a threshold (default 0.05). This is the
Kennel-Brown-Abarbanel heuristic.

## Public API

```python
from kuant.sindy.chaos import falsenearest

fnn = falsenearest(x, tau=1, max_dim=10, r_tol=15.0, threshold=0.05)
print(fnn.summary())
m = fnn.suggested_m
```

Signature:

```python
falsenearest(x, *, tau=1, max_dim=10, r_tol=15.0, threshold=0.05)
```

Returns `FalseNearestResult` with `dims`, `fnn` (fraction per dim),
`suggested_m`, and the `threshold` used.

## Design decisions

### `r_tol=15.0` default

Kennel et al. recommend a tolerance ratio in the range 10 to 30 for
the "extra coordinate outgrows the current nearest-neighbor
distance" criterion. 15 is the conventional middle of that band.

### 5% FNN threshold

`suggested_m` is the first `m` where `fnn[m] <= threshold`. Default
0.05. Falls back to `max_dim` when no dimension crosses the
threshold, usually a sign that the series is too noisy or too short
for a clean FNN signature. Inspect the raw `fnn` curve before
trusting the fallback.

### Brute-force O(N^2) nearest-neighbor search

For each candidate dimension, pairwise squared distances are
computed in a full `(N, N)` matrix. Correct and simple; adequate
for the up-to-a-few-thousand embedded points typical here. A
KD-tree variant would be faster but would add a scipy dependency
the rest of `kuant.sindy.chaos` avoids.

### 100-observation floor

Below 100 finite values, the FNN fraction is dominated by
finite-sample noise and the kernel rejects the call.

### Zero-distance guard

When a nearest-neighbor distance is exactly zero (duplicate
embeddings, common on discrete-valued series), the pair is scored
as "true" rather than dividing by zero to produce a false-neighbor
verdict.

## When it fires

- Step two of the standard `(tau, m)` picking workflow, after
  `mutualinfo`.
- Called internally by `chaosscan` to pick `m` at the auto-picked
  `tau` before running the rest of the battery.

## References

- Kennel, Brown & Abarbanel 1992, "Determining embedding dimension
  for phase-space reconstruction using a geometrical construction."

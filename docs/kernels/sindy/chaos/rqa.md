# rqa: Marwan-Romano-Thiel-Kurths recurrence quantification

## Purpose

Compute the recurrence plot of a time-delay embedded series and
distill it into five scalars:

- **Recurrence rate (RR)**: fraction of `(i, j)` pairs whose
  embedded distance is at or below `epsilon`. Density of the plot.
- **Determinism (DET)**: fraction of recurrent pairs lying on
  diagonal segments of length at least `l_min`. High DET signals
  deterministic dynamics; near-zero DET signals stochastic noise.
- **Laminarity (LAM)**: the same fraction, on vertical segments.
  Distinguishes chaotic switching from laminar or trapped states.
- **Longest diagonal**: inverse-proportional to the divergence rate,
  a Lyapunov proxy.
- **Entropy of diagonal lengths**: Shannon entropy (nats) of the
  diagonal-length distribution. Peaks for complex dynamics.

## Public API

```python
from kuant.sindy.chaos import rqa

r = rqa(x, tau=1, m=5, recurrence_rate_target=0.1, l_min=2)
print(r.summary())
```

Signature:

```python
rqa(
    x, *, tau=1, m=5,
    epsilon=None, recurrence_rate_target=0.1, l_min=2,
)
```

Returns `RQAResult` with all five measures plus the `epsilon` and
`l_min` actually used, and `embed_dim`, `embed_tau`.

## Design decisions

### `epsilon` auto-pick from a target recurrence rate

If `epsilon` is `None`, the kernel picks the quantile of off-diagonal
pairwise distances that hits `recurrence_rate_target` (default 0.10).
Absolute `epsilon` values are not comparable across series with
different scales; a target recurrence rate is, so the auto-pick is
the safe default when comparing plots across signals.

### N capped at 2000

The recurrence matrix is `N x N` in memory (dense `int8`). Series
longer than 2000 observations are truncated to the last 2000 rows
before embedding. That caps peak memory at a few megabytes for the
matrix itself and keeps the diagonal and vertical scans in bounded
time. Callers who need older data should window explicitly.

### `l_min=2` default

Minimum diagonal or vertical length counted toward DET / LAM.
Length-1 recurrences are single-pixel events; counting them would
let noise-driven flicker dominate both statistics.

### Line-of-identity excluded

The main diagonal (all `(i, i)`) is zeroed before computing RR,
DET, LAM, and the longest-diagonal statistic. It carries no
dynamical information and would inflate every scalar.

### 100-observation floor

Below 100 finite values, the diagonals and verticals are too short
for stable DET / LAM.

## When it fires

- When Lyapunov and correlation dimension leave the regime
  ambiguous. RQA reads structure the other two miss (intermittency,
  laminar phases, transition points).
- Called by `chaosscan` for the classifier's DET input (the
  chaotic verdict requires `DET > 0.5`; the periodic verdict
  requires `DET > 0.9`).

## References

- Marwan, Romano, Thiel & Kurths 2007, "Recurrence plots for the
  analysis of complex systems."

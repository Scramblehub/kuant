# corrdim: Grassberger-Procaccia correlation dimension

## Purpose

Estimate the correlation dimension `D_2` of a time-delay embedded
attractor. `C(r)`, the fraction of embedded-point pairs closer than
`r`, scales as `C(r) ~ r^D_2` over the middle band of the log-log
curve. The kernel fits that slope.

`D_2` is the go-to low-dim-chaos-vs-noise discriminant:

- Deterministic low-dim chaos: `D_2` is a small non-integer that
  saturates once `m` exceeds the true attractor dimension (e.g.
  ~2.05 for the Lorenz attractor).
- Periodic dynamics: `D_2 ~ 1` (trajectory lies on a 1D closed
  curve).
- Stochastic noise: `D_2` grows with `m` without saturating.

Run `corrdim` at several `m` and inspect the saturation curve. The
saturation-vs-`m` behavior is the actual test, not the single-`m`
scalar.

## Public API

```python
from kuant.sindy.chaos import corrdim

cd = corrdim(x, tau=1, m=5, n_r=20, r_frac_range=(0.05, 0.5))
print(cd.summary())
cd.correlation_dim
cd.log_r, cd.log_C       # full log-log curve for visual inspection
```

Signature:

```python
corrdim(x, *, tau=1, m=5, n_r=20, r_frac_range=(0.05, 0.5))
```

Returns `CorrDimResult` with `correlation_dim`, `log_r`, `log_C`,
`fit_range`, `embed_dim`, `embed_tau`.

## Design decisions

### Fit the middle 60% of the log-log curve

`fit_range = [0.2 * n, 0.8 * n]` of the surviving log-log points.
The small-`r` tail is dominated by individual-pair discreteness
(noise floor), and the large-`r` tail saturates at the attractor
diameter (finite-size cutoff). Only the middle band respects the
power law, so that is where the linear fit lives.

### `r_frac_range=(0.05, 0.5)` default

Radii are log-spaced between 5% and 50% of the largest observed
pairwise distance. Values outside that band consistently fall into
one of the two above pathologies.

### Full pairwise-distance matrix

The kernel builds the full `(N, N)` distance matrix and reads the
upper triangle. O(N^2) time and memory. Acceptable in the 300 to
~a-few-thousand observation regime; longer inputs should be windowed
before being passed in.

### 300-observation floor

Below 300 finite values, the upper-triangle pair count is too small
to fill even 20 log-spaced radii bins reliably.

### Saturation-vs-`m` is the chaos discriminant

A single `D_2` number is not diagnostic. Run `corrdim` at
`m = 3, 4, 5, 6, 7` and read the sequence: saturation to a fixed
value is the chaotic signature; monotone growth is the noise
signature. The composer `chaosscan` uses `D_2 >= embed_dim - 0.5`
as its "didn't saturate" proxy.

## When it fires

- The correlation-dimension half of the chaotic-vs-noise decision,
  complementing the Lyapunov test.
- Called by `chaosscan` for the regime classifier's `D_2` input.

## References

- Grassberger & Procaccia 1983, "Characterization of strange
  attractors."

# chaosscan: Composer for the chaos battery + regime label

## Purpose

One call, one regime label. Given a 1D series:

1. Pick `tau` from the first minimum of the auto-mutual-information
   curve (`mutualinfo`).
2. Pick `m` from the false-nearest-neighbors threshold crossing
   (`falsenearest`) at that `tau`.
3. Run `lyapunov`, `corrdim`, and `rqa` at the picked `(tau, m)`.
4. Apply a rule-based classifier to the three scalars and label the
   regime as one of `{chaotic, periodic, stochastic, unknown}`.

Cross-kernel consistency is the point of the composer: every
primitive uses the same `_embed(x, m, tau)`, so the three scalars
fed into the classifier all live on the same reconstructed
attractor.

## Public API

```python
from kuant.sindy.chaos import chaosscan

res = chaosscan(x)                        # fully auto
res = chaosscan(x, tau=3, m=6)            # user-picked (tau, m)

print(res.regime)                         # 'chaotic' | 'periodic' | ...
print(res.summary())

# Raw per-kernel results are attached for inspection.
res.mutualinfo, res.falsenearest
res.lyapunov, res.corrdim, res.rqa
```

Signature:

```python
chaosscan(x, *, tau=None, m=None, max_lag=32, max_dim=10, n_r=20)
```

Returns `ChaosScanResult` with `regime`, the picked `embed_tau` and
`embed_dim`, and each underlying kernel's full result dataclass.

## Design decisions

### Rule-based classifier thresholds

The `_classify` rules, evaluated in order:

- **chaotic**: `lyapunov > 0.001` AND `d2 < embed_dim - 0.5` AND
  `det > 0.5`. Positive divergence, sub-embedding-dimension
  attractor, diagonal structure in the recurrence plot.
- **periodic**: `abs(lyapunov) < 0.005` AND `d2 < 1.5` AND
  `det > 0.9`. Near-zero divergence, near-1D geometry,
  almost-fully-deterministic recurrence.
- **stochastic**: `d2 >= embed_dim - 0.5` AND `det < 0.5`. No
  saturation in correlation dimension, no diagonal structure.
- **unknown**: anything else.

These are the thresholds the literature settles on for
well-conditioned test signals. Callers who disagree can pull the
raw kernel results off the returned dataclass and apply their own
rules; the label is a convenience, not the ground truth.

### `unknown` is a real label

`unknown` fires when the three underlying signals disagree. Typical
cases:

- Positive Lyapunov but high `D_2` (noisy chaos, or `m` picked too
  low so the attractor is not fully unfolded).
- Near-zero Lyapunov but low DET (colored noise, or a periodic
  signal with `epsilon` set too tight).
- Lyapunov and DET agree on "chaotic" but `D_2` did not saturate
  (`d2 >= embed_dim - 0.5`).

When `unknown` fires, inspect `lyapunov.log_divergence`,
`corrdim.log_r / log_C`, and `rqa.determinism` before either
re-picking `(tau, m)` manually or accepting the ambiguity.

### 300-observation floor

Bounded by `corrdim`'s minimum. `chaosscan` rejects short series
before running any primitive.

### CCM is not part of the battery

`chaosscan` scans a single series. For nonlinear causality between
two series, call `ccm` directly.

## When it fires

- The default entry point for "is my series chaotic?" workflows.
- Roll it per window (with fixed `tau` and `m`, to hold the
  reconstruction constant) as a regime-shift monitor: the label
  transitions are what matter, not the absolute values.
- If the label comes back `unknown`, inspect the per-kernel
  results before re-picking `(tau, m)`.

## References

- Fraser & Swinney 1986, "Independent coordinates for strange
  attractors from mutual information."
- Kennel, Brown & Abarbanel 1992, "Determining embedding dimension
  for phase-space reconstruction using a geometrical construction."
- Rosenstein, Collins & DeLuca 1993, "A practical method for
  calculating largest Lyapunov exponents from small data sets."
- Grassberger & Procaccia 1983, "Characterization of strange
  attractors."
- Marwan, Romano, Thiel & Kurths 2007, "Recurrence plots for the
  analysis of complex systems."

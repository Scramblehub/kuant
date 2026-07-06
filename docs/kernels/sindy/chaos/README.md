# kuant.sindy.chaos

Chaos-theory diagnostics for time-series. Distinguish deterministic
low-dimensional chaos from noise, quantify recurrence structure, and
test nonlinear causality — all with numpy-only, CPU-only kernels that
compose through a single time-delay embedding.

## What's in the subpackage

Six primitives plus a composer:

1. **`mutualinfo`** ([`mutualinfo.md`](mutualinfo.md)) — Fraser-Swinney
   1986 histogram MI. Two modes: auto-MI curve vs lag (picks embedding
   delay `tau` at the first local minimum), or cross-MI scalar between
   two series at a given lag.
2. **`falsenearest`** ([`falsenearest.md`](falsenearest.md)) —
   Kennel-Brown-Abarbanel 1992 false-nearest-neighbors. Picks the
   smallest embedding dimension `m` at which the attractor is fully
   unfolded (FNN fraction drops below a threshold).
3. **`lyapunov`** ([`lyapunov.md`](lyapunov.md)) — Rosenstein-Collins-
   DeLuca 1993 largest Lyapunov exponent. Positive lambda = chaotic
   fingerprint. Returns the full log-divergence curve so users can eye
   the linear-fit region.
4. **`corrdim`** ([`corrdim.md`](corrdim.md)) — Grassberger-Procaccia
   1983 correlation dimension `D_2`. Fits the middle 60% of the log-log
   pair-count curve. Saturates for chaos; grows with `m` for noise.
5. **`rqa`** ([`rqa.md`](rqa.md)) — Marwan-Romano-Thiel-Kurths 2007
   recurrence quantification. Recurrence rate, determinism, laminarity,
   longest diagonal, entropy of diagonal lengths. Auto-picks epsilon
   to hit a target recurrence rate if not supplied.
6. **`ccm`** ([`ccm.md`](ccm.md)) — Sugihara 2012 convergent
   cross-mapping. Tests nonlinear causality between two series by
   asking whether one's shadow manifold predicts the other's future,
   with prediction skill rising as the reference library grows.
7. **`chaosscan`** ([`chaosscan.md`](chaosscan.md)) — composer. Auto-
   picks `(tau, m)`, runs the full battery, and classifies into
   `{chaotic, periodic, stochastic, unknown}` with rule-based
   thresholds.

## Shared internals

All six primitives share one time-delay embedding helper
(`_embed(x, m, tau)`), so any consistent choice of `(tau, m)` is
directly comparable across the battery.

## When to use what

| Question | Kernel |
|---|---|
| What's my embedding delay `tau`? | `mutualinfo(x)` |
| What's my embedding dim `m` at that tau? | `falsenearest(x, tau=tau)` |
| Is the trajectory chaotic (positive lambda)? | `lyapunov` |
| How low-dimensional is the attractor? | `corrdim` |
| How much recurrence structure is there? | `rqa` |
| Does `x` causally drive `y`? | `ccm(x, y)` |
| Just tell me the regime label | `chaosscan(x)` |

## Financial notes

- Financial return series are typically **stochastic** by this
  battery's metrics (D_2 doesn't saturate, DET is low). This is a
  finding, not a bug: it means you cannot get away with treating them
  as low-dim attractors.
- **Regime residuals** (returns minus a fitted HMM/regime-model mean)
  are more useful subjects. The `chaosscan` regime label plus
  `ccm` between residual streams is a natural stack for regime-shift
  monitoring.
- All kernels reject fewer than a few hundred finite observations to
  keep estimates stable. For the CCM and correlation-dimension
  kernels, the minimum is 200-300 clean rows.

## Cross-kernel identities enforced in tests

- **Periodic sinusoid**: `corrdim < 2`, `rqa.determinism` or
  `rqa.laminarity > 0.7`, `chaosscan` regime not `chaotic`.
- **Gaussian noise**: `corrdim` grows with `m` (no saturation);
  `chaosscan` regime in `{stochastic, unknown}`.
- **r=4 logistic map**: `lyapunov > 0` (positive Lyapunov signature).
- **Independent Gaussians**: `ccm` should not show convergence in
  both directions simultaneously.

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
- Sugihara et al 2012, "Detecting causality in complex ecosystems."

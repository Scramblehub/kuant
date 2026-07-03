# posteriorentropy — Shannon entropy of an HMM posterior per bar

## Purpose

For each time step `t`, compute the Shannon entropy of the posterior
distribution over hidden states:

```math
H[t] = -Σ_i γ[t, i] · log γ[t, i]
```

Range `[0, log N]`. Low → posterior is CONFIDENT (concentrated on one
state). High → posterior is DIFFUSE (near-uniform).

Real-world finding: entropy-weighted gating often outperforms
threshold-based gating on BOTH Sharpe and MDD. Rationale: the model was
usually right about the current state, but sometimes so uncertain that
acting on the prediction was noise-driven.

## Public API

```python
from kuant.qm import posteriorentropy

result = posteriorentropy(gamma, regime=None)
print(result.summary())
```

- `gamma` — `(T, N)` HMM posterior (from `kuant.qm.hmm.posterior` or
  `kuant.qm.ghmm.posterior`)
- `regime` — optional `(T,)` categorical indicator. If provided,
  per-regime entropy statistics are attached.

Returns `PosteriorEntropyResult` with `entropy` (T,), `max_entropy`
(= `log N`), and optional `per_regime` breakdown.

## Design decisions

### 0·log(0) handled cleanly

`xp.where(gamma > 0, gamma·log(gamma), 0)`. Prevents NaN when a state's
posterior is exactly zero.

### Backend-preserving

numpy in → numpy out; cupy in → cupy out. `per_regime` stats are
computed on the CPU-side entropy array via a one-time `.get()` if
input is cupy.

### Per-regime entropy — a QM-analog pattern to look for

In stress regimes (however you label them for your target), a good
HMM's posterior tends to COLLAPSE — entropy drops sharply because the
model becomes confident about which state applies. In calm regimes
the same posterior often blurs — entropy near the uniform bound
because state boundaries are ambiguous. When you observe this pattern
in your own data, entropy-weighted gating is likely to outperform
threshold gating; if you don't, the model isn't distinguishing
regimes cleanly and gating on posterior confidence probably won't
help.

## Related tools

- `kuant.qm.hmm.posterior` — produces the input `γ`
- `kuant.qm.ghmm.posterior` — same, for continuous-emission HMM
- `kuant.qm.zenoscan` — sibling scan for retrain-frequency choice

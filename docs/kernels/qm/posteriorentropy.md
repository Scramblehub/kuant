# posteriorentropy — Shannon entropy of an HMM posterior per bar

## Purpose

For each time step `t`, compute the Shannon entropy of the posterior
distribution over hidden states:

```math
H[t] = -Σ_i γ[t, i] · log γ[t, i]
```

Range `[0, log N]`. Low → posterior is CONFIDENT (concentrated on one
state). High → posterior is DIFFUSE (near-uniform).

Production win in our V8 research: entropy-weighted gating outperformed
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

### Per-regime entropy — the observed pattern

In our V8 HMM data:

| Regime label | Mean entropy | Interpretation |
|---|---|---|
| High-VIX | 0.180 | Posterior collapses — model is confident |
| Low-VIX | 0.373 | Posterior blurs — model is uncertain |

Reads like a QM analog: "measurement" (stress) collapses the wave
function; calm regimes leave it in superposition.

## Related tools

- `kuant.qm.hmm.posterior` — produces the input `γ`
- `kuant.qm.ghmm.posterior` — same, for continuous-emission HMM
- `kuant.qm.zenoscan` — sibling scan for retrain-frequency choice

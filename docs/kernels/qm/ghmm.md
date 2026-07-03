# kuant.qm.ghmm — Gaussian-observation HMM inference

Same four algorithms as `kuant.qm.hmm` (forward / backward / viterbi /
posterior), but observations are **continuous scalars** and per-state
emissions are `N(μ_i, σ_i²)`.

This is what V8 regime work actually uses in production — returns are
continuous, not discrete symbols.

## API

```python
from kuant.qm.ghmm import forward, backward, viterbi, posterior

log_alpha, log_lik  = forward(obs, pi, A, mu, sigma)
log_beta            = backward(obs, pi, A, mu, sigma)
states, log_prob    = viterbi(obs, pi, A, mu, sigma)
gamma, xi, log_lik  = posterior(obs, pi, A, mu, sigma)
```

**Inputs (all four):**

- `obs` — 1D float array of scalar observations, length `T`
- `pi` — 1D array, initial state distribution, shape `(N,)`
- `A` — transition matrix, shape `(N, N)`, rows sum to 1
- `mu` — per-state emission mean, shape `(N,)`
- `sigma` — per-state emission std, shape `(N,)`, all **> 0**

The `obs`/`pi`/`A` semantics are identical to discrete HMM; only the
emission model changes:

```math
B[t, i] = (1/√(2π·σ_i²)) · exp(-(obs[t]-μ_i)² / (2σ_i²))
```

## Shared setup

`kuant/qm/ghmm/common.py::_prepare_ghmm_inputs` validates all inputs
and precomputes `log_pi`, `log_A`, `log_B` (shape `(T, N)`, in
log-space directly to skip an unnecessary `log(exp(...))`).

`_LOG_SQRT_2PI = 0.5 · log(2π)` is cached at module load.

## Design decisions

### Log-space emission likelihood

Computed as:

```math
log B[t, i] = -½·log(2π) - log σ_i - ½·((obs[t]-μ_i)/σ_i)²
```

Numerically stable across many orders of magnitude of `sigma`. No
underflow issues on long sequences or extreme observations.

### `sigma > 0` guard

Sanity-checked in `_prepare_ghmm_inputs`. Zero or negative σ raises
`ValueError` — otherwise the log-space math would produce `+inf` or
`NaN`.

### Reuses `_logsumexp_axis` from discrete hmm

The scalar recursions are structurally identical between discrete and
Gaussian HMM — only `B` changes. Rather than duplicate, ghmm imports
the shared logsumexp helper from `kuant.qm.hmm.forward`.

## Tests (5)

- Forward/backward likelihood match
- Viterbi recovers regime when means are well-separated
- Posterior γ rows sum to 1, ξ slices sum to 1
- Negative sigma raises
- Shape mismatch (`mu`, `sigma`, `A` vs `pi`) raises

## Related tools

- `kuant.qm.hmm` — discrete-observation counterpart
- `kuant.qm.posteriorentropy` — apply to ghmm's γ output for confidence
- `kuant.qm.zenoscan`, `kuant.qm.decoherencescan` — retrain-frequency
  and within-window scans, work with either HMM flavor via
  `fit_fn` / `predict_fn` callables

# kuant.qm.hmm — Hidden Markov model inference (discrete observations)

Log-space implementations of the four standard HMM algorithms.

## API

```python
from kuant.qm.hmm import forward, backward, viterbi, posterior

log_alpha, log_lik   = forward(obs, pi, A, B)
log_beta             = backward(obs, pi, A, B)
states, log_prob     = viterbi(obs, pi, A, B)
gamma, xi, log_lik   = posterior(obs, pi, A, B)
```

**Inputs (all four):**

- `obs` — 1D int array of observation indices in `[0, M)`, length `T`
- `pi` — 1D array, initial state distribution, shape `(N,)`
- `A` — transition matrix, shape `(N, N)`, rows sum to 1
- `B` — emission matrix, shape `(N, M)`, rows sum to 1

## Algorithms

### `forward(obs, pi, A, B) → (log_alpha, log_likelihood)`

Log-space forward recursion:

```math
α[0, i]   = π[i] · B[i, o_0]
α[t, j]   = Σ_i α[t-1, i] · A[i, j] · B[j, o_t]
```

Returns `log_alpha` of shape `(T, N)` and the sequence log-likelihood
`log P(O | model) = logsumexp_i(log_alpha[T-1, i])`.

### `backward(obs, pi, A, B) → log_beta`

Log-space backward recursion:

```math
β[T-1, i] = 1
β[t, i]   = Σ_j A[i, j] · B[j, o_{t+1}] · β[t+1, j]
```

Returns `log_beta` of shape `(T, N)`. The `pi` argument is unused
mathematically but validated for API symmetry.

### `viterbi(obs, pi, A, B) → (states, log_prob)`

Most-likely hidden state sequence via dynamic programming. Log-space
throughout with an explicit backpointer table `ψ`. Returns:

- `states` — 1D int array of length `T`, values in `[0, N)`
- `log_prob` — log-probability of the returned path

### `posterior(obs, pi, A, B) → (gamma, xi, log_likelihood)`

Uses `forward` and `backward` to compute:

```math
γ[t, i]    = P(s_t = i | O)     = α[t, i] · β[t, i] / P(O)
ξ[t, i, j] = P(s_t = i, s_{t+1} = j | O)
```

Returns `gamma` shape `(T, N)` with rows summing to 1, `xi` shape
`(T-1, N, N)` with per-`t` slices summing to 1, and the same
`log_likelihood` as `forward`.

## Design decisions

### Log-space throughout

`scipy.special.logsumexp` on numpy; inline stable logsumexp on cupy.
Prevents underflow on long sequences without needing scaling factors.

### `zeno_scan`-friendly renormalization on `gamma`, `xi`

After exponentiating log-space quantities, we renormalize each row of
`gamma` and each slice of `xi` to sum to 1 exactly. Defends against
tiny FP drift so downstream `zenoscan` / Baum-Welch code doesn't need
to.

### Cross-verified sanity

- `forward` log-likelihood matches `backward` sanity computation
- `gamma[t]` equals row-marginal of `xi[t]` (both computed independently)
- `viterbi` respects transition costs (test with `A ≈ I` locks into
  state given a strong prior)

## Tests (13)

Forward shapes and single-observation base case, backward base row,
forward/backward likelihood match, Viterbi with strong prior + strong
emissions, gamma rows sum to 1, xi slices sum to 1, gamma marginalizes
xi, posterior log-likelihood matches forward.

## Related tools

- `kuant.qm.belltest` — test whether the HMM's joint posterior beats
  classical aggregation of the same features
- `kuant.qm.zenoscan` — test retrain-frequency effect on HMM (or any
  model) skill

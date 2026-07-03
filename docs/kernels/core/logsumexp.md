# logsumexp — Numerically stable log-sum-exp

## Purpose

Compute `log(sum(exp(x)))` without intermediate overflow. Classic
numerical stability trick:

```math
\text{logsumexp}(x) = m + \log\left(\sum_i \exp(x_i - m)\right)
                    \quad\text{where}\quad m = \max_i x_i
```

Subtracting the max prevents `exp(x_i)` from overflowing to `inf`.

## Public API

```python
from kuant.core import logsumexp

result = logsumexp(x, axis=None, keepdims=False)
```

- `x` — scalar or array.
- `axis` — optional axis or tuple of axes to reduce over. Default
  reduces all axes.
- `keepdims` — preserve reduced dims as size-1.

## Uses

- **HMM forward-backward** (`kuant.qm`) — combining log-alpha values
- **Bayesian model averaging** — normalizing log-posteriors
- **Information-theoretic tests** — log-space probability aggregation
- Any log-probability workflow

## Design decisions

### -inf handling

If all elements in a reduced slice are `-inf`, `max` is `-inf`,
`m_safe` gets replaced with 0 for the shift, `exp(x - 0) = 0` for all,
`sum = 0`, `log(0) = -inf`, and `-inf + (-inf) = -inf`. Correct
result.

The intermediate `log(0)` triggers a numpy divide-by-zero warning
that is *cosmetic* — the result value is the correct `-inf`. We
suppress that warning with `np.errstate(divide='ignore')`.

### Backend-preserving

Uses `xp.max`, `xp.sum`, `xp.exp`, `xp.log` throughout. cupy input
→ cupy output.

### int input promoted to float64

Prevents surprising behavior when someone passes integer log-probs.

## Related

- `scipy.special.logsumexp` — reference behavior
- `kuant.qm.hmm` — heavy consumer for forward-backward

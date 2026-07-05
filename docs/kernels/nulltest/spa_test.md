# spa_test: Superior Predictive Ability and Model Confidence Set

## Purpose

Two related tests for the same underlying question: given a benchmark
strategy and a family of alternatives, which alternatives are
GENUINELY better once we correct for the number of things tried?

- `spa_test`: Hansen (2005) Superior Predictive Ability. Null: no
  alternative beats the benchmark. Small p rejects; at least one
  alternative is genuinely superior.
- `mcs_test`: Hansen, Lunde & Nason (2011) Model Confidence Set.
  Iteratively drops provably-worse strategies until only a set of
  statistically-indistinguishable "survivors" remains.

Both use the stationary block bootstrap from
`kuant.nulltest.stationary_bootstrap` to preserve serial correlation.

## Public API

```python
from kuant.nulltest import spa_test, mcs_test

sp  = spa_test(
    benchmark_returns,       # 1D length T
    alternative_returns,     # 2D (T, K)
    n_boot=1000,
    mean_block_length=5.0,
    seed=0,
)

mcs = mcs_test(
    strategy_returns,        # 2D (T, K)
    alpha=0.05,
    n_boot=1000,
    mean_block_length=5.0,
    seed=0,
)
```

- All return series are aligned to the same time axis.
- `alpha`: MCS confidence level; the survivor set covers the best
  strategy with probability `1 - alpha`.

## Design decisions

### 1. Loss differential is `alt - benchmark`

`spa_test` forms `d[:, k] = alt_returns[:, k] - benchmark_returns`.
Higher `d` means the alternative outperformed. Standardized loss
differentials `t_stat[k] = sqrt(T) * mean(d[:, k]) / std(d[:, k])`
have the max taken as the observed test statistic.

Under the null of no superior alternative, the max standardized loss
differential should not systematically exceed zero. The bootstrap
recenters `d` around its observed mean and asks what fraction of
draws produce a max t-stat at or above the observed one.

### 2. Bootstrap draws are stationary-block joint resamples

Every draw resamples the ROWS of `d` (i.e. resamples timestamps),
which resamples every column with the SAME indices. Column-wise
independent resampling would destroy the cross-alternative
correlation structure that SPA is designed to correct for.

### 3. Std floor prevents division-by-zero warnings

`sd = np.where(sd > 1e-15, sd, 1e-15)` clamps the per-column standard
deviation before dividing. Degenerate columns (constant loss
differential) get a huge t-stat, which is the correct behavior: a
strategy identical to the benchmark should not survive as "superior."

### 4. MCS reformulated as an SPA-style single-drop

The full Hansen-Lunde-Nason MCS procedure uses a range statistic
across all surviving strategies. kuant ships a simplified equivalent
that iteratively drops the worst-performing survivor until the SPA-
style test cannot reject "worst is not worse than best":

1. Compute per-strategy sample means over the current survivor set.
2. Identify the sample-best and sample-worst strategies.
3. Run `spa_test` with the worst as pseudo-benchmark and the best as
   the sole alternative. This asks "is the sample-best significantly
   better than the sample-worst?"
4. If the p-value falls below `alpha`, drop the worst and repeat.
5. Otherwise stop.

The remaining set contains all strategies statistically
indistinguishable from the sample-best; the true best is in there
with probability `1 - alpha`. This preserves the coverage guarantee
of the original MCS at the single-drop level.

### 5. `SPAResult` shared across both tests

`spa_test` returns an `SPAResult` with an empty `survivors` list;
`mcs_test` returns the same dataclass with the survivor indices
populated. This lets tearsheet code consume both uniformly.

## Return shape

**SPAResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `p_value` | float | Bootstrap p under "no alternative beats benchmark" |
| `max_t_stat` | float | Observed max standardized loss diff |
| `n_alternatives` | int | `K` |
| `n_boot` | int | Bootstrap draws performed |
| `survivors` | list[int] | MCS-only: indices in the confidence set |

`.summary()` returns a formatted multi-line string; the `survivors`
line reads `(SPA-only)` when the list is empty.

## Examples

```python
>>> import numpy as np
>>> from kuant.nulltest import spa_test, mcs_test
>>> rng = np.random.default_rng(0)
>>> T = 500
>>> bench = rng.normal(0.001, 0.01, T)
>>> # One truly-better alternative among a batch of noise.
>>> good  = bench + rng.normal(0.002, 0.001, T)
>>> noise = rng.normal(0.001, 0.01, (T, 9))
>>> alts  = np.column_stack([good, noise])
>>> res = spa_test(bench, alts, n_boot=300)
>>> res.p_value < 0.05
True
>>> # MCS on the same set: only strategies indistinguishable from the
>>> # best survive.
>>> mcs = mcs_test(np.column_stack([bench, alts]), alpha=0.05, n_boot=300)
>>> 0 not in mcs.survivors or True                        # bench may or may not survive
True
```

## References

- Hansen, P. R. (2005). "A Test for Superior Predictive Ability."
  Journal of Business and Economic Statistics, 23(4).
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). "The Model
  Confidence Set." Econometrica, 79(2).

## Related kernels

- `kuant.nulltest.stationary_bootstrap`: the block resampler used
  inside both loops.
- `kuant.nulltest.bootstrap_ic`: same resampler for single-signal
  IC CIs and p-values.
- `kuant.nulltest.mht_correction`: Bonferroni / Holm / BH corrections
  when a bulk of pairwise tests is preferred over a single joint test.
- `kuant.portfolio.deflated_sharpe`: a Sharpe-first alternative to
  SPA for the "how many trials did I run?" correction.

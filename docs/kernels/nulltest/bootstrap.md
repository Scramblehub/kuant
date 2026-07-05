# bootstrap: Block bootstrap for serially-correlated series

## Purpose

Standard i.i.d. bootstrap breaks on time-series data because return
sequences carry serial correlation. This module ships two block
resampling primitives that preserve short-range dependence, plus a
convenience wrapper that produces a full IC confidence interval and
two-sided p-value in one call.

- `stationary_bootstrap`: one Politis-Romano resample of a 1D series.
- `bootstrap_ic`: block-bootstrap CI + p-value for a signal's IC
  against forward returns.

## Public API

```python
from kuant.nulltest import stationary_bootstrap, bootstrap_ic

sample = stationary_bootstrap(series, mean_block_length=5, seed=0)
result = bootstrap_ic(
    signal,
    forward_returns,
    n_boot=1000,
    mean_block_length=5.0,
    seed=0,
)
```

- `series`, `signal`, `forward_returns`: 1D arrays. Length >= 2 for
  the raw resampler; `bootstrap_ic` requires >= 10 clean (finite)
  rows after aligning `(signal, forward_returns)`.
- `mean_block_length`: expected block size. Set larger for series
  with stronger serial correlation. Rule of thumb: `T^(1/3)` for
  weakly-dependent series.
- `n_boot`: number of bootstrap draws.
- `seed`: RNG seed (`numpy.random.default_rng`).

## Design decisions

### 1. Stationary bootstrap over moving block

The moving-block bootstrap (Kunsch 1989) uses fixed-length blocks
drawn with replacement, which produces a non-stationary resample.
The stationary bootstrap (Politis & Romano 1994) uses block lengths
drawn from a geometric distribution with mean `mean_block_length`;
the resulting resampled series is itself strictly stationary, which
matters for downstream statistics that assume stationarity.

The one-line implementation: start at a random index; at each step,
either extend the current block (probability `1 - p`) or jump to a
new random index (probability `p`), where `p = 1 / mean_block_length`.

### 2. Wraparound at the tail

When extending a block past the end of the series, we wrap:
`i = (i + 1) % n`. Wraparound preserves the geometric block-length
distribution at the boundary and is the standard Politis-Romano
convention.

### 3. `bootstrap_ic` resamples JOINTLY

The point of a bootstrap IC is to preserve the `(signal, return)`
CORRELATION under resampling. That means both series must be
resampled with the SAME block indices at every draw. `bootstrap_ic`
generates one index array per draw and applies it to both cleaned
arrays; drawing them independently would destroy the correlation and
give useless confidence intervals.

### 4. Fast Pearson inside the loop

The bootstrap loop needs to run `n_boot * n` operations. Rather than
call out to `scipy.stats.spearmanr` on every draw, `bootstrap_ic`
uses an inline Pearson correlation. If you need bootstrap intervals
on a Spearman IC, run `factor_ic` for the per-period series and
bootstrap the resulting 1D array with `stationary_bootstrap`.

### 5. Two-sided p-value under the IC = 0 null

Given an observed point IC, the p-value asks: under the null of zero
true IC, how often would we see a bootstrap draw as-or-more-extreme
in the opposite direction?

```python
if point > 0:
    p = 2 * fraction(boot <= 0)
elif point < 0:
    p = 2 * fraction(boot >= 0)
else:
    p = 1.0
p = min(p, 1.0)
```

This is the wrong-sign-tail two-sided test used in Politis-Romano
applications. The 95% CI is the empirical [2.5%, 97.5%] percentile
range of the bootstrap distribution.

### 6. Clean-row requirement

Non-finite rows in either `signal` or `forward_returns` are dropped
before resampling. Fewer than 10 clean rows raises
`KuantValueError` (`KE-VAL-MIN-CLEAN`); running a bootstrap on 5
observations produces a distribution that is uniform on 5 points
and gives meaningless confidence bands.

## Return shape

`stationary_bootstrap` returns a 1D `np.ndarray` of the same length
and dtype as `series`.

**BootstrapICResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `point_estimate` | float | IC on the un-resampled data |
| `bootstrap_distribution` | 1D array, len `n_boot` | Per-draw IC |
| `p_value` | float | Two-sided p-value under IC = 0 |
| `ci_low` | float | 2.5th percentile of bootstrap ICs |
| `ci_high` | float | 97.5th percentile of bootstrap ICs |
| `n_boot` | int | Draws performed |
| `mean_block_length` | float | Expected block size used |

`.summary()` returns a formatted multi-line string.

## Examples

```python
>>> import numpy as np
>>> from kuant.nulltest import stationary_bootstrap, bootstrap_ic
>>> rng = np.random.default_rng(0)
>>> x = rng.standard_normal(200)
>>> sample = stationary_bootstrap(x, mean_block_length=5, seed=42)
>>> sample.shape == x.shape
True
>>> sig = rng.standard_normal(500)
>>> ret = 0.1 * sig + 0.3 * rng.standard_normal(500)     # real IC ~0.3
>>> res = bootstrap_ic(sig, ret, n_boot=300)
>>> res.p_value < 0.05                                    # signal detected
True
>>> res.ci_low < res.point_estimate < res.ci_high
True
```

## References

- Kunsch, H. R. (1989). "The Jackknife and the Bootstrap for General
  Stationary Observations." Annals of Statistics, 17(3).
- Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
  Journal of the American Statistical Association, 89(428).

## Related kernels

- `kuant.nulltest.spa_test`, `kuant.nulltest.mcs_test`: multi-strategy
  null tests that use the same stationary-block resampler.
- `kuant.nulltest.mht_correction`: Bonferroni / Holm / BH correction
  when you run the bootstrap over many candidate signals.
- `kuant.signals.factor_ic`: Spearman IC series to feed into
  `stationary_bootstrap` when a rank-based bootstrap CI is needed.

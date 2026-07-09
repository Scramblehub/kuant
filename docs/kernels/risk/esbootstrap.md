# esbootstrap — Bootstrap Expected Shortfall with confidence interval

## Purpose

Expected Shortfall (ES) is the average loss GIVEN that the loss
exceeds VaR. Because ES is a tail statistic, its sample estimate has
high variance, and a bare ES point number is easy to over-read.
`esbootstrap` returns the point estimate together with a
bootstrap CI, quantifying that noise.

Uses a moving-block bootstrap (Kunsch 1989) so that any short-range
serial dependence in returns is preserved inside each resampled
block. Set `block_size = 1` to fall back to iid bootstrap.

## Public API

```python
from kuant.risk import esbootstrap

result = esbootstrap(
    returns,
    conf_alpha=0.95,
    ci_alpha=0.95,
    n_boot=500,
    block_size=21,
    seed=0,
)
print(result.summary())
print(result.es_point, result.es_ci_low, result.es_ci_high)
```

- `returns`: 1D array-like. Non-finite entries stripped.
- `conf_alpha`: VaR / ES confidence level in `[0.5, 0.9999]`. Default
  `0.95`.
- `ci_alpha`: bootstrap CI level in `[0.5, 0.9999]`. Default `0.95`.
- `n_boot`: number of bootstrap replicates. Default `500`.
- `block_size`: moving-block length. Default `21` (approximately one
  trading month at daily cadence). Must be `<= n / 2`.
- `seed`: PRNG seed for reproducibility. Default `0`.

Returns `EsBootstrapResult` with fields `es_point`, `es_ci_low`,
`es_ci_high`, `var_point`, `ci_alpha`, `conf_alpha`, `n_boot`,
`block_size`, `n`.

## Design decisions

### 1. ES from empirical tail, not from a fitted distribution

```math
\text{VaR}_\alpha = Q_\alpha(-r), \quad
\text{ES}_\alpha  = \mathbb{E}[-r \mid -r \geq \text{VaR}_\alpha]
```

Implemented as `losses[losses >= q].mean()`. No parametric
assumption. If the tail is heavy enough that the empirical ES is
untrustworthy on its own, escalate to `evtvar`; this kernel
quantifies noise, not model risk.

### 2. Moving-block bootstrap (Kunsch 1989)

Each replicate draws `ceil(n / block_size)` random start indices,
concatenates the contiguous blocks, and truncates to length `n`.
That preserves within-block autocorrelation (volatility clustering,
serial dependence). Block length is a bias-variance knob: too short
loses dependence structure; too long inflates variance across
replicates. `21` (a trading month) is a reasonable default at daily
cadence; adjust to the autocorrelation scale of the input series.

Politis-Romano 1994 (stationary bootstrap) is the natural
alternative with random block length; queued for a later version if
users ask for it. Set `block_size = 1` for the classical iid
bootstrap of Efron.

### 3. Percentile CI, not BCa

CI endpoints are `[Q_{(1-ci_alpha)/2}, Q_{1-(1-ci_alpha)/2}]` of the
bootstrap ES distribution. Percentile-method: simple, no bias /
acceleration parameters. Good enough when the bootstrap ES
distribution is approximately symmetric. If the input tail is
extreme enough to skew that distribution meaningfully, that itself
is a signal to switch to `evtvar`.

### 4. Positive-loss sign convention

Same as the rest of `kuant.risk`. `es_point >= var_point >= 0` in
the normal case (ES averages a tail with heavier losses than the
threshold). The test suite asserts this ordering.

### 5. Minimum sample

`n_finite < 100` raises `KE-VAL-MIN-CLEAN`. Fewer than 100 points is
not enough for ES at reasonable `conf_alpha` (a 95% tail is only 5
observations at `n = 100`).

### 6. `block_size` range guard

`block_size > n // 2` raises `KE-VAL-RANGE`. With larger blocks the
resampled series is dominated by one or two contiguous chunks and
bootstrap variance collapses artificially.

## Edge cases

| Condition | Behavior |
| --- | --- |
| Empty tail (`losses >= q` empty, degenerate) | ES equals VaR for that replicate |
| `block_size == 1` | iid bootstrap |
| `block_size > n // 2` | `KuantValueError` with `KE-VAL-RANGE` |
| `n_finite < 100` | `KuantValueError` with `KE-VAL-MIN-CLEAN` |
| `conf_alpha` or `ci_alpha` out of range | `KuantValueError` with `KE-VAL-RANGE` |
| `n_boot <= 0` or `block_size <= 0` | `KuantValueError` |
| Non-finite entries | stripped before resampling |

## Cross-check tests

- `test_esbootstrap_ci_contains_point`: CI brackets `es_point`, and
  `es_point >= var_point`.
- `test_esbootstrap_shrinks_ci_with_more_data`: 500 vs 5000 points,
  same seed and blocks. CI width strictly shrinks with `n`.
- `test_esbootstrap_block_size_range_error`: `block_size = 200` on
  `n = 200` raises.
- `test_min_clean_gate` (parametrized): fewer than 100 finite values
  raises with `KE-VAL-MIN-CLEAN`.

## Direct usage in kuant

Wrap around any historical ES report. If a monitored ES ticks up by
a small amount and the tick sits inside the CI band, that is noise.
If the tick sits outside the CI computed on the pre-tick window, it
is signal.

## Related kernels

- `kuant.risk.evtvar`: parametric alternative for heavy-tailed
  series. Complementary: `esbootstrap` quantifies sampling noise;
  `evtvar` extrapolates into the tail.
- `kuant.risk.cornishfishervar`: closed-form VaR for near-Gaussian
  series.

## References

- Cont, R., Deguest, R., Scandolo, G. 2010. "Robustness and
  sensitivity analysis of risk measurement procedures."
  *Quantitative Finance*.
- Kunsch, H. 1989. "The jackknife and the bootstrap for general
  stationary observations." *Annals of Statistics*.
- Politis, D., Romano, J. 1994. "The stationary bootstrap." *Journal
  of the American Statistical Association*.

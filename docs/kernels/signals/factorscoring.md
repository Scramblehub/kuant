# factorscoring: Cross-sectional factor evaluation

## Purpose

Given a `(T, N)` panel of factor scores and an aligned `(T, N)` panel
of forward returns, quantify the factor's ability to rank names.

Ships five kernels:

- `factor_ic`: per-period cross-sectional Information Coefficient.
- `factor_rank_autocorr`: how sticky the factor's ranking is.
- `mean_return_by_quantile`: per-period per-bucket forward return.
- `quantile_spread`: top-bucket-minus-bottom-bucket return series.
- `quantile_turnover`: churn in the top and bottom buckets.

Cross-sectional (compare N names at each date). For time-series
univariate signal vs forward return, see `kuant.signals.icdecay`.

## Public API

```python
from kuant.signals import (
    factor_ic,
    factor_rank_autocorr,
    mean_return_by_quantile,
    quantile_spread,
    quantile_turnover,
)

ic     = factor_ic(factor, forward_returns, method='spearman')
racorr = factor_rank_autocorr(factor, lag=1)
qret   = mean_return_by_quantile(factor, forward_returns, n_quantiles=5)
qspr   = quantile_spread(factor, forward_returns, n_quantiles=5)
qturn  = quantile_turnover(factor, n_quantiles=5)
```

- `factor`, `forward_returns`: 2D arrays of shape `(T, N)`, dates on
  rows, names on columns. Same shape.
- `method`: `'spearman'` (default), `'pearson'`, or `'kendall'`.
- `lag`: positive int; `factor_rank_autocorr` compares row `t` to
  row `t - lag`.
- `n_quantiles`: int in `[2, 100]`.

All five require `scipy.stats` for the rank correlations.

## Design decisions

### 1. Cross-sectional at every date, then aggregate across time

At each date `t`, take the N-vector of factor scores and the N-vector
of forward returns for the same N names, compute the correlation, and
store `ic[t]`. Aggregate statistics (mean IC, IR, t-stat) come from
that length-`T` series, never from a pooled `(T*N,)` regression.

Pooled regressions bake in cross-sectional heteroskedasticity and
period-scale differences (e.g. a high-vol month dominates). The
per-date average is the standard factor-research convention.

### 2. NaN policy: row-by-row, not panel-wide

A NaN in one name at one date should not disqualify the whole date's
correlation or the whole name's history. Each row builds its own
finite-pair mask and computes the correlation on those pairs. Rows
with fewer than 3 finite pairs yield NaN for that period; those NaNs
are dropped from the aggregate mean, std, IR, and t-stat.

### 3. IR uses raw `sqrt(T)`: annualization is your problem

`ir = mean_IC / std_IC * sqrt(n_periods)`. We do not multiply by a
frequency factor. If your panel is daily and you want an annualized
IR, multiply by `sqrt(252 / step)` yourself; if it is monthly,
`sqrt(12)`. Baking a frequency in would silently corrupt users on
weekly or other cadences.

### 4. Quantile bucketing uses `np.quantile` at each row

For each date, `np.quantile` on that date's finite scores gives
`n_quantiles - 1` interior cut points; `np.digitize` maps each finite
score to a bin in `[0, n_quantiles)`. NaN cells get bin `-1` and are
excluded from that row's per-bucket mean.

Rows with fewer than `n_quantiles` finite values are entirely skipped
(a 3-name row cannot support 5 buckets).

### 5. `quantile_spread` reuses `mean_return_by_quantile`

`quantile_spread` is defined as top-bucket mean minus bottom-bucket
mean, per period. Rather than duplicate the bucketing logic, it calls
`mean_return_by_quantile` and subtracts columns `Nq-1` and `0`.
The t-stat is `mean / (std / sqrt(n))` where `n` is the count of
finite spread values.

### 6. Turnover measures departures, not arrivals

`top_turnover[t]` is the fraction of yesterday's top-bucket names
that FELL OUT of today's top bucket. Symmetric definition for the
bottom bucket. Position 0 is NaN by construction. Interpretation:
higher turnover implies more transaction-cost drag on any strategy
that rotates by the factor.

## Return shape / dataclass fields

**FactorICResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `ic` | 1D array, len T | Per-period IC; NaN if <3 pairs |
| `mean` | float | Mean of the finite ICs |
| `std` | float | Sample std of finite ICs (ddof=1) |
| `ir` | float | `mean / std * sqrt(n_periods)` |
| `t_stat` | float | `mean / (std / sqrt(n_periods))` |
| `n_periods` | int | Count of finite ICs |
| `method` | str | Correlation method used |

**RankAutocorrResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `autocorr` | 1D array, len T | Row-t vs row-(t-lag) Spearman |
| `mean` | float | Mean across finite entries |
| `lag` | int | Lag used |
| `n_periods` | int | Count of finite entries |

**QuantileReturnsResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `mean_by_quantile` | 2D array `(T, Nq)` | Bucket mean forward return |
| `total_by_quantile` | 1D array, len Nq | Column-wise nansum |
| `n_quantiles` | int | Buckets used |

**QuantileSpreadResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `spread` | 1D array, len T | Top-minus-bottom mean return |
| `mean`, `std`, `t_stat` | float | Aggregate stats over finite spread |
| `n_periods`, `n_quantiles` | int | |

**QuantileTurnoverResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `top_turnover` | 1D array, len T | Fraction of top-Q dropouts |
| `bottom_turnover` | 1D array, len T | Fraction of bot-Q dropouts |
| `top_mean`, `bottom_mean` | float | Aggregate churn |
| `n_periods`, `n_quantiles` | int | |

Every result has a `.summary()` that returns a formatted string.

## Examples

```python
>>> import numpy as np
>>> from kuant.signals import factor_ic, quantile_spread
>>> rng = np.random.default_rng(0)
>>> T, N = 500, 50
>>> f = rng.standard_normal((T, N))
>>> r = 0.05 * f + rng.standard_normal((T, N))          # 5% IC by construction
>>> ic = factor_ic(f, r)
>>> ic.mean > 0.02
True
>>> qs = quantile_spread(f, r, n_quantiles=5)
>>> qs.mean > 0                                          # top out-earns bottom
True
>>> abs(qs.t_stat) > 2                                   # significant
True
```

## Related kernels

- `kuant.signals.icdecay`: time-series univariate IC-vs-horizon curve.
- `kuant.signals.winsorize`, `kuant.signals.neutralize`: cross-
  sectional pre-processing that typically runs before this family.
- `kuant.nulltest.bootstrap_ic`: block-bootstrap p-value and CI on
  an IC point estimate.

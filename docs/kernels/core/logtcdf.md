# logtcdf — Numerically stable log of Student-t CDF

## Purpose

Compute `log(tcdf(x, df))` without underflow in the deep left tail.

Naive `log(tcdf(x, df))` underflows to `-inf` at extreme negative x
where `tcdf(x, df)` rounds to 0 in float64. This kernel adds an
asymptotic tail fallback that keeps the result finite.

## Public API

```python
from kuant.core import logtcdf

result = logtcdf(x, df)
```

Returns `log(tcdf(x, df))`. Always finite for valid inputs; `NaN`
for `NaN` x, `NaN` df, or `df ≤ 0`.

## Design decisions

### Two-branch computation

- **Normal range**: `log(tcdf(x, df))` computed directly. Stable for
  `|x|` up to ~1e100 (the underflow limit depends on df).
- **Deep-tail fallback**: when the direct value underflows to 0 (only
  triggered at extreme negative x), use leading-order series of the
  regularized incomplete beta:
  ```
  log(tcdf(x, df)) ≈ log(1/2) + a·log(z) - log(a) - lnB(a, 1/2)
  ```
  where `a = df/2` and `z = df/(df + x²)`. Absolute error `O(z)` in
  the tail.

### Warning suppression

Inactive branches may evaluate `log(0)` under vectorization even
though `where` masks them out. Suppressed via
`np.errstate(divide='ignore', invalid='ignore')`.

## Uses

- Fat-tail log-likelihood computations
- Extreme-quantile probability estimation
- POT / EVT tail scoring
- Bayesian model comparison with heavy-tailed likelihoods

## Related

- `kuant.core.tcdf` — the CDF being log-transformed
- `kuant.core.logtccdf` — upper-tail companion
- `kuant.core.lognormcdf` — Gaussian analog
- `scipy.stats.t.logcdf` — reference implementation

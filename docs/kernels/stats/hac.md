# hac — Heteroskedasticity-and-autocorrelation-consistent standard errors

## Purpose

Two HAC covariance estimators for the linear model `y = X beta + u` with
serially correlated, heteroskedastic errors:

- `neweywestse` (Newey-West 1987): Bartlett kernel. The most-cited HAC.
- `andrewsse` (Andrews 1991): quadratic-spectral (QS) kernel with an
  AR(1) plug-in bandwidth. Better small-sample properties than
  Newey-West, at the cost of a slower log-log kernel weight loop.

Both share the same OLS pre-solve and return `HacResult` with `beta`,
`se`, full `cov`, residuals, and the kernel bandwidth actually used.

## Public API

```python
from kuant.stats import neweywestse, andrewsse

r_nw  = neweywestse(y, X)             # auto bandwidth = floor(4 * (n/100)^(2/9))
r_qs  = andrewsse(y, X, bandwidth=8)  # or specify explicitly
```

- `y` — 1D array, length `n`. Cast to float64.
- `X` — 2D array, shape `(n, k)`. Include a leading ones column for an
  intercept.
- `bandwidth` — positive int, optional. `None` triggers the automatic
  rule per kernel.
- Returns `HacResult(beta, se, cov, residuals, n, k, bandwidth, kernel)`.
  `HacResult.summary()` prints beta, SE, and t-stats.

## Design decisions

### 1. Shared pre-solve; kernel differs only in the middle

Both kernels compute `beta = (X'X)^{-1} X'y` once, form the score
contributions `u_i = e_i * X_i`, then wrap them in

```math
Cov(beta) = (X'X)^{-1} S (X'X)^{-1}
```

where `S` is the kernel-weighted long-run variance of the scores. Only
`S` differs across kernels, so the OLS path and error handling are
shared in `_check_xy` and `_ols`.

### 2. Newey-West: Bartlett kernel, linearly decaying weights

Weight at lag `L` is `w_L = 1 - L / (bandwidth + 1)` for `L = 1..bw`.
Guarantees positive semi-definite `S`. Auto-bandwidth follows the
Newey-West 1994 rule `floor(4 * (n / 100) ^ (2/9))`, clamped to at
least 1. Loop cost is `O(bw * k^2)`; cheap even at large `n`.

### 3. Andrews QS: AR(1) plug-in bandwidth, kernel truncated by weight

Andrews 1991 Table 1 gives the optimal bandwidth for the QS kernel as
`1.3221 * (alpha_2 * n) ^ (1/5)` where `alpha_2 = 4 rho^2 / (1 - rho)^4`
and `rho` is the AR(1) coefficient of the summed scores. `rho` is
clipped to `[-0.97, 0.97]` to avoid the divergent tail.

The QS kernel decays as `~ 1 / z^2 * (sin - cos)`; we sum lags up to
`min(n - 1, 3 * bandwidth)` and skip lags where `|w| < 1e-8`. That cut
covers effectively all mass without the full `n - 1` sweep.

### 4. Non-finite rows dropped, then `n >= k + 10` enforced

`_check_xy` drops rows with any NaN or Inf in `y` or `X`, then requires
at least `k + 10` clean rows. Below that OLS is not identified and the
HAC middle has too few autocovariances to be meaningful:

```
KE-VAL-MIN-CLEAN: "after dropping non-finite rows, {n} rows and {k}
regressors; need at least k+10 rows."
```

### 5. Bandwidth floor of 1

A zero bandwidth reduces the estimator to White heteroskedasticity-only
SEs, which is not what the caller asked for. `max(int(bandwidth), 1)`
enforces the floor before the require-positive check.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `y.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-EXPECTED` |
| `X.ndim != 2` | raises `KuantShapeError` `KE-SHAPE-EXPECTED` |
| `X.shape[0] != y.size` | raises `KuantShapeError` `KE-SHAPE-EQUAL-LEN` |
| Fewer than `k + 10` finite rows | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| `bandwidth <= 0` | clamped to 1 |
| `rho` near 1 in Andrews plug-in | clipped to 0.97 |
| Any NaN/Inf in a row | that row is dropped |

## Cross-check tests

- `test_neweywest_positive_se` — SEs strictly positive; betas within
  0.2 of true (0.5, 0.3) on 300-point synthetic AR-free data
- `test_andrews_positive_se` — Andrews SEs strictly positive on the
  same fixture
- `test_bad_shape_rejected` — mismatched `y` and `X` lengths raise
  `KuantShapeError`
- `test_summary` — `HacResult.summary()` renders

## References

- Newey, W. K., West, K. D. (1987). "A simple positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix."
  Econometrica 55, 703-708.
- Newey, W. K., West, K. D. (1994). "Automatic lag selection in
  covariance matrix estimation." Review of Economic Studies 61,
  631-653.
- Andrews, D. W. K. (1991). "Heteroskedasticity and autocorrelation
  consistent covariance matrix estimation." Econometrica 59, 817-858.

## Related

- `kuant.stats.autocorrtests.ljungbox` — check whether HAC is needed
  (portmanteau on OLS residuals)
- `kuant.stats.autocorrtests.durbinwatson` — first-order smoke check
- `kuant.stats.stationarity` — pre-tests before running OLS

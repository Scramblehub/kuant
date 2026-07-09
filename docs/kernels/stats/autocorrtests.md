# autocorrtests — Portmanteau and first-order autocorrelation tests

## Purpose

Three residual-diagnostic staples for time-series and regression work,
all testing the null of no autocorrelation:

- `ljungbox` (Ljung-Box 1978): portmanteau across lags `1..h`. The
  default reported alongside GARCH / OLS output.
- `boxpierce` (Box-Pierce 1970): the original portmanteau. Lower
  small-sample power than Ljung-Box; kept for legacy comparison.
- `durbinwatson`: first-order autocorrelation smoke check. Fast,
  bounded in `[0, 4]`, near 2 under the null.

## Public API

```python
from kuant.stats import ljungbox, boxpierce, durbinwatson

lb = ljungbox(residuals, h=10, dof_correction=0)
bp = boxpierce(residuals, h=10)
dw = durbinwatson(residuals)
```

- `x` — 1D array of residuals or a stationary series. Non-finite
  values are dropped.
- `h` — number of autocorrelation lags. Must be in `[1, n - 1]`.
- `dof_correction` — subtract for ARMA-fitted residuals (p + q). Must
  be `< h`.
- Returns `PortmanteauResult(stat, p_value, h, dof, test)` or
  `DurbinWatsonResult(stat, n)`.

## Design decisions

### 1. Shared ACF core

`_acf(x, max_lag)` demeans once, computes the denominator `sum(x*x)`
once, then loops over lags with the sample autocovariance formula.
Both portmanteau tests reuse it; the only difference is how they
weight `acf[1:]^2`:

```
LB   = n * (n + 2) * sum(acf[k]^2 / (n - k))     # small-sample correction
BP   = n * sum(acf[k]^2)                          # unweighted
```

Both are chi-square under H0 with `dof = h - dof_correction`.

### 2. `_chi2_sf` with a scipy-free fallback

Primary path uses `scipy.stats.chi2.sf`. Fallback is the
Wilson-Hilferty transform:

```
z = ((x/df)^(1/3) - (1 - 2/(9 df))) / sqrt(2/(9 df))
p = 1 - Phi(z)
```

Accurate to about 3 digits in the right tail, which is what portmanteau
tests care about. No hard dependency on scipy at import time.

### 3. Non-finite drop then `n >= 20` floor

Any NaN/Inf entries are removed before the sample size check. A
minimum of 20 clean values is required across all three tests:

```
KE-VAL-MIN-CLEAN: "only {n} finite values; need at least 20."
```

Below 20 the chi-square approximation is unreliable and the ACF
denominator can collapse to noise.

### 4. `dof_correction` guarded against `dof <= 0`

If the caller passes `dof_correction >= h` the chi-square distribution
has no degrees of freedom left:

```
KE-VAL-RANGE: "dof_correction ({dof_correction}) must be < h ({h})."
```

Typical usage: `h = 20, dof_correction = p + q` for an ARMA(p, q) fit.

### 5. Durbin-Watson: no p-value

DW's null distribution depends on `X` (regressor matrix), so no
distribution-free p-value is reported. `summary()` prints a text
interpretation:

- `< 1.5` -> positive first-order autocorrelation
- `1.5..2.5` -> no first-order autocorrelation
- `> 2.5` -> negative first-order autocorrelation

For a formal test the caller should look up the DW critical bounds
`d_L, d_U` for `(n, k)` in Savin-White (1977) or use the Ljung-Box
alternative here.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| Fewer than 20 finite values | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| `h < 1` or `h > n - 1` | raises `KuantValueError` `KE-VAL-RANGE` |
| `dof_correction >= h` | raises `KuantValueError` `KE-VAL-RANGE` |
| DW denominator ~ 0 | returns `stat = NaN` |
| Constant series | ACF denominator ~ 0; ACF returns zeros; stat ~ 0 |

## Cross-check tests

- `test_ljungbox_iid_not_significant` — 500-point Gaussian noise:
  p > 0.05
- `test_ljungbox_ar1_significant` — AR(1) with rho = 0.7: p < 0.01
- `test_boxpierce_matches_shape` — stat > 0 on 300-point noise
- `test_durbinwatson_iid_near_two` — DW in `(1.7, 2.3)` for iid
- `test_durbinwatson_ar1_low` — DW < 1.0 for AR(1) rho = 0.8
- `test_too_short_rejected` — 10-point input raises `KuantValueError`

## References

- Ljung, G. M., Box, G. E. P. (1978). "On a measure of lack of fit in
  time series models." Biometrika 65, 297-303.
- Box, G. E. P., Pierce, D. A. (1970). "Distribution of residual
  autocorrelations in autoregressive-integrated moving average time
  series models." JASA 65, 1509-1526.
- Durbin, J., Watson, G. S. (1950, 1951). "Testing for serial
  correlation in least squares regression." Biometrika 37 and 38.
- Savin, N. E., White, K. J. (1977). "The Durbin-Watson test for serial
  correlation with extreme sample sizes or many regressors."
  Econometrica 45, 1989-1996.

## Related

- `kuant.stats.hac.neweywestse` — remediation when ljungbox rejects on
  regression residuals
- `kuant.stats.stationarity` — pre-tests before portmanteau
- `kuant.stats.bdstest` — nonlinear iid test; catches structure that
  portmanteau misses when linear ACF is zero

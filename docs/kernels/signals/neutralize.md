# neutralize — OLS residual against factor exposures

## Purpose

Regress a signal on one or more factor exposure series and return the
residual:

```
signal_t = alpha + sum_k beta_k * factor_k(t) + residual_t
```

The residual is the "factor-neutralized signal": the part of the
signal that is not explained by the factors. Used to strip incidental
exposures out of a raw alpha, so a long-short portfolio built from it
does not accidentally tilt into value / size / momentum / market beta
simply because the raw signal correlated with those factors.

## Public API

```python
from kuant.signals import neutralize

result = neutralize(signal, factors={'size': size_ts, 'value': value_ts})
result.residuals               # 1D, len T; NaN where any input was NaN
result.betas                   # {'intercept': ..., 'size': ..., 'value': ...}
result.r2                      # variance fraction explained
result.condition_number        # of (X.T @ X); collinearity diagnostic
result.n_used                  # rows the fit ran on after NaN drop
```

- `signal`: 1D array, length `T`.
- `factors`: one of
  - 2D array `(T, K)`, names default to `factor0`, `factor1`, ...
  - `dict[str, 1D array]`, keys become factor names
  - `list` of 1D arrays, names default to `factor0`, ...
- `add_intercept`: bool, default `True`. Adds a column of ones. Setting
  `False` forces regression through zero (only appropriate when the
  signal is already centered).

Example:

```python
>>> import numpy as np
>>> from kuant.signals import neutralize
>>> rng = np.random.default_rng(0)
>>> factor = rng.standard_normal(500)
>>> signal = 0.5 * factor + rng.standard_normal(500) * 0.1
>>> r = neutralize(signal, {'factor': factor})
>>> abs(r.betas['factor'] - 0.5) < 0.05
True
>>> r.r2 > 0.9
True
```

## Design decisions

### 1. `np.linalg.lstsq`, SVD-based

Preferred over the normal-equation solution `beta = (X.T X)^-1 X.T y`
because SVD is stable when `X` is near-singular. `rcond=None` uses
the numpy default cutoff.

### 2. Row-mask NaN drop before fitting

`row_mask = isfinite(signal) & isfinite(X).all(axis=1)`. Any row with
a NaN in the signal or any factor is dropped from the fit; the
corresponding output residual is NaN. That is the "PIT-clean" choice,
factor exposures at a bar with no signal do not participate in
learning `beta`.

If fewer clean rows survive than the design has parameters, the
regression is underdetermined and we raise `KuantValueError` with
`[KE-VAL-UNDERDET]` rather than returning an arbitrary least-norm
solution.

### 3. Condition-number diagnostic + collinearity warning

We compute `cond = (s_max / s_min) ** 2` from the SVD of the clean
design. That is the condition number of `X^T X` (equivalently, the
squared condition number of `X`). Above `1e10` we emit
`KuantNumericWarning` with `[KW-COLLINEAR-FACTORS]`. Above that
threshold, factor betas swing widely under tiny perturbations and
should not be interpreted individually.

Zero smallest singular value maps `cond` to `inf`.

### 4. Constant-signal `KW-NEUTRALIZE-CONSTANT-SIGNAL` warning

`ss_tot = 0` means the signal has no variance across the fit rows;
`R^2` is undefined. We report `r2 = 0` and emit the warning rather
than either raising or returning NaN silently.

### 5. Three factor input forms

`dict`, 2D array, and list are all accepted so callers can pass what
they have. Named coefficients (from dict keys, or `factor0` /
`factor1` from the positional forms) show up in `result.betas` for
readability.

### 6. `betas` includes `intercept` iff `add_intercept=True`

The intercept coefficient shares the dict with factor coefficients;
`design_names[0] == 'intercept'` when the intercept is present. Users
who need the pure factor-only exposures should read
`{k: v for k, v in result.betas.items() if k != 'intercept'}`.

## Return shape

**NeutralizeResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `residuals` | 1D array, len `T` | Fit residuals; NaN on dropped rows |
| `betas` | dict | Coefficient per column of the design matrix |
| `r2` | float | `1 - SSR / SST` on the fit rows |
| `condition_number` | float | `(s_max / s_min) ** 2` of clean design |
| `n_used` | int | Rows after NaN drop |

## Edge cases

| Condition | Behavior |
| --- | --- |
| `factors` not dict / array / list | `KuantValueError` `[KE-SHAPE-EXPECTED]` |
| `factors` array with `ndim > 2` | `KuantShapeError` `[KE-SHAPE-EXPECTED]` |
| `factors` empty | `KuantValueError` `[KE-VAL-RANGE]` |
| Length mismatch | `KuantValueError` via `require_equal_length` |
| `n_used < X.shape[1]` after NaN drop | `KuantValueError` `[KE-VAL-UNDERDET]` |
| `cond > 1e10` | `KuantNumericWarning` `[KW-COLLINEAR-FACTORS]` |
| `ss_tot == 0` | `KuantNumericWarning` `[KW-NEUTRALIZE-CONSTANT-SIGNAL]`, `r2 = 0` |

## Related kernels

- `kuant.signals.winsorize`: standard preprocessing upstream, to keep
  outliers from dominating the OLS fit.
- `kuant.signals.icdecay`: run on the residuals to check the
  neutralized signal still has forecasting power.
- `kuant.signals.whitening`: covariance-level analogue when factors
  are the columns of a design matrix and you want the full
  decorrelation, not a single-signal residual.

## References

- Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management:
  A Quantitative Approach for Producing Superior Returns and
  Controlling Risk*, 2nd ed. McGraw-Hill.
- Rosenberg, B. (1974). Extra-market components of covariance in
  security returns. *Journal of Financial and Quantitative Analysis*,
  9(2), 263-274.

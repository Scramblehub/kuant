# rdd — Sharp regression discontinuity via local linear regression

## Purpose

Estimates the causal jump in outcome `Y` at a threshold `cutoff` in a
running variable `x`. Sharp design: treatment is deterministic in `x`,
`D = 1[x >= cutoff]`. Under smoothness of `E[Y | X]` on either side of
the cutoff, the discontinuity at the cutoff IS the local average
treatment effect.

Applies to program-eligibility cutoffs, index-inclusion thresholds,
rating-boundary events, and any assignment mechanism keyed to a
continuous score.

## Public API

```python
from kuant.causal import rdd

res = rdd(x, y, cutoff=0.0)
print(res.summary())
print(res.tau, res.tau_se, res.tau_t_stat)
```

- `x` : 1D array. Running variable.
- `y` : 1D array, same length as `x`. Outcome.
- `cutoff` : float. Threshold; treatment applies at `x >= cutoff`.
- `bandwidth` : float, optional. Half-width around `cutoff`. If `None`,
  uses the simple rule-of-thumb `1.5 * std(x) * n^{-0.2}`.

Returns `RddResult` with `tau`, `tau_se`, `tau_t_stat`, `n_left`,
`n_right`, `intercept_left`, `intercept_right`, `slope_left`,
`slope_right`, `bandwidth`, `cutoff`.

## Design decisions

### 1. Triangular kernel, weighted least squares per side

Standard non-parametric RDD estimator: two local linear regressions,
one on each side of the cutoff, both weighted by the triangular
kernel:

```math
K(u) = \max(1 - |u|, 0), \quad u = (x - c) / h
```

Full weight at the cutoff, zero at `+/- h`. Imbens-Kalyanaraman 2012
show this kernel is MSE-optimal for the local linear estimator among
kernels supported on `[-1, 1]`.

### 2. `tau` sign convention

```text
tau = intercept_right - intercept_left
    = E[Y | x -> cutoff+] - E[Y | x -> cutoff-]
```

Positive `tau` means the outcome JUMPS UP at the cutoff. The left- and
right-side slopes are reported separately for diagnostic plotting.

### 3. Independent SE per side, then combined by variance sum

`tau_se = sqrt(se_a_left^2 + se_a_right^2)`. Both sides are estimated
from disjoint samples, so their intercept estimators are independent
and their variances add. This is the standard RDD SE convention.

### 4. Degrees of freedom use unweighted count `n - 2`

WLS in `_wls` uses `dof = max(n - 2, 1)` (statsmodels convention),
NOT the kernel-weighted effective sample size. Calonico-Cattaneo-
Titiunik 2014 use the same convention for the point-estimate SE
(their bias correction is separate machinery). This keeps `tau_se`
comparable to published RDD tables.

### 5. Default bandwidth is a rule of thumb, not IK-optimal

`h = 1.5 * std(x) * n^{-0.2}`. Roughly matches Silverman-style
bandwidths and gives sane defaults for smoke tests. Production use
should pass an Imbens-Kalyanaraman 2012 or Calonico-Cattaneo-Titiunik
2014 optimal bandwidth explicitly. The API surfaces `bandwidth` on
the result so callers can iterate.

### 6. Minimum-per-side guardrail of 5 observations

If fewer than 5 clean observations land on either side of the cutoff
inside the bandwidth, raises `[KE-VAL-MIN-CLEAN]` with a message
telling the caller to widen the bandwidth. Five is the minimum for a
non-degenerate two-parameter WLS with a residual.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x` and `y` different length | raises `[KE-SHAPE-EQUAL-LEN]` |
| Non-1D `x` or `y` | raises via `require_1d` |
| Fewer than 40 clean rows overall | raises `[KE-VAL-MIN-CLEAN]` |
| Fewer than 5 on left or right within bandwidth | raises `[KE-VAL-MIN-CLEAN]` |
| `bandwidth <= 0` (user-supplied) | raises via `require_positive` |
| `tau_se == 0` (degenerate fit) | `tau_t_stat` returned as `inf` |
| Non-finite `x` or `y` rows | dropped before estimation |

## Cross-check tests

- `test_rdd_recovers_true_jump` : DGP with a true 2.5 jump at
  `cutoff = 0`, `n = 3000`. Recovers `tau` within 0.15 with more than
  20 observations on each side.
- `test_rdd_no_jump_null` : DGP with no jump. `|tau| < 0.15`.
- `test_rdd_shape_error` : mismatched length raises.
- `test_rdd_min_clean_error` : `n = 10` raises the 40-row minimum.

## Direct usage in kuant

Index-inclusion event studies (Russell reconstitution, S&P rebalance),
credit-rating boundary studies (BB+ vs BB), and any cutoff-based
policy or eligibility jump on a continuous score. Composes with
`synthcontrol` for hybrid designs where a cutoff exists but the treated
side is a single unit.

## Related kernels

- [`iv`](iv.md) : the fuzzy-RDD variant is IV with the cutoff dummy as
  the instrument. Not implemented as a separate kernel yet: call `iv`
  directly with `Z = 1[x >= cutoff]`.
- [`synthcontrol`](synthcontrol.md) : the "one unit, many controls"
  alternative when no threshold exists.

## References

- Imbens, G. W., Lemieux, T. (2008). "Regression Discontinuity Designs:
  A Guide to Practice." *Journal of Econometrics* 142(2). Framework
  and estimator.
- Imbens, G. W., Kalyanaraman, K. (2012). "Optimal Bandwidth Choice for
  the Regression Discontinuity Estimator." *ReStud* 79(3). MSE-optimal
  bandwidth and triangular-kernel result.
- Calonico, S., Cattaneo, M. D., Titiunik, R. (2014). "Robust
  Nonparametric Confidence Intervals for Regression-Discontinuity
  Designs." *Econometrica* 82(6). Modern bias-corrected variant and
  the SE degrees-of-freedom convention used here.

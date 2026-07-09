# iv — Two-stage least squares instrumental variables

## Purpose

Estimates the causal effect of endogenous regressor(s) `X` on outcome
`Y` using instrument(s) `Z` that (i) correlate with `X` (relevance) and
(ii) affect `Y` only through `X` (exclusion). Addresses the endogeneity
bias that plagues naive OLS when `X` is co-determined with unobservables
in the error term.

Just-identified case: `dim(Z) == dim(X)`. Over-identified:
`dim(Z) > dim(X)` (Sargan test not implemented in v0.6 scope).

## Public API

```python
from kuant.causal import iv

# Single endogenous regressor with one instrument.
res = iv(y, x_endog, z_instr)
print(res.summary())
print(res.beta, res.se, res.t_stat, res.f_stat_stage1)
```

- `y` : 1D array of length `n`. Outcome.
- `x_endog` : 2D array `(n, k_endog)`. Endogenous regressor(s). A 1D
  column is auto-reshaped.
- `z_instr` : 2D array `(n, k_instr)`. Instrument(s). Requires
  `k_instr >= k_endog`.
- `add_intercept` : bool, default `True`. Prepends a column of ones to
  both stages; the reported `beta` is trimmed of the intercept.

Returns `IvResult` with `beta`, `se`, `t_stat`, `f_stat_stage1`,
`r2_stage1`, `r2_stage2`, `n`, `k_endog`, `k_instr`.

## Design decisions

### 1. Sigma^2 uses residuals against original X, not X_hat

The subtle correctness point in 2SLS. After stage 2 gives
`beta_2 = (X_hat' X_hat)^{-1} X_hat' y`, the variance is:

```text
sigma^2 = sum((y - X @ beta_2)^2) / (n - k)     # against ORIGINAL X
Var(beta_2) = sigma^2 * (X_hat' X_hat)^{-1}
```

Using `y - X_hat @ beta_2` in the residual instead systematically
understates `sigma^2` (and hence SE / t-stats), since `X_hat` is the
projection of `X` onto the instrument space and absorbs less variance.
Wooldridge Chapter 5 and Hayashi Chapter 3.5 both document the
correction. The kernel implements it explicitly.

### 2. Stage-1 F on excluded instruments

For the single-endogenous case (`k_endog == 1`), stage-1 F is computed
from stage-1 R^2:

```math
F = \frac{R^2}{1 - R^2} \cdot \frac{n - k_{instr} - 1}{k_{instr}}
```

Under `F < 10`, warns via `KW-IV-WEAK-INSTRUMENT` referencing
Staiger-Stock 1997. Weak instruments bias 2SLS toward OLS AND make the
sampling distribution non-normal, so both the point estimate and its
SE are unreliable in that regime.

For multi-endogenous (`k_endog > 1`), the scalar F is reported as
`nan`; the correct diagnostic is the Cragg-Donald or
Kleibergen-Paap min-eigenvalue statistic (out of scope for v0.6).

### 3. Auto-transpose of 1D-shaped inputs

If `x_endog` or `z_instr` come in as `(1, n)` where `n == y.size`,
they are auto-transposed to `(n, 1)`. Prevents a common shape mistake
without silently mis-fitting a wrong-orientation matrix.

### 4. Clean-row masking with a minimum of `k_endog + k_instr + 5`

Rows with any non-finite entry in `y`, `x_endog`, or `z_instr` are
dropped. The minimum clean count of `k_endog + k_instr + 5` guarantees
positive degrees of freedom for both stages plus a small buffer for a
non-degenerate F.

### 5. `pinv` on the second-stage Gram matrix

`np.linalg.pinv(X_hat' X_hat)` rather than `inv`. Handles near-singular
Gram matrices without raising, at a small cost that is invisible next
to `lstsq` on stage 1.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `k_instr < k_endog` (under-identified) | raises `[KE-VAL-RANGE]` |
| Length mismatch across `y`, `x_endog`, `z_instr` | raises `[KE-SHAPE-EQUAL-LEN]` |
| Fewer than `k_endog + k_instr + 5` clean rows | raises `[KE-VAL-MIN-CLEAN]` |
| Stage-1 F < 10, single endog | warns `KW-IV-WEAK-INSTRUMENT` |
| Stage-1 F undefined for multi-endog | reports `nan`, no warning |
| Non-2D `x_endog` or `z_instr` after auto-reshape | raises via `require_2d` |

## Cross-check tests

- `test_iv_recovers_true_beta_when_ols_is_biased` : DGP with
  `y = 2 + 3x + u + 0.6v` and `x = 0.8z + v`. OLS would be biased
  upward by 0.6; IV recovers `beta = 3.0` within 0.2 and reports
  stage-1 F > 50.
- `test_iv_underidentified_error` : 2 endog, 1 instr raises
  `KuantValueError`.
- `test_iv_weak_instrument_warning` : first-stage coefficient 0.02
  triggers `KW-IV-WEAK-INSTRUMENT`.

## Direct usage in kuant

Any factor-return regression where the factor is plausibly endogenous
with the error term: shift-share instruments, lagged-innovation
instruments, and policy-shock instruments all fit this API. Also the
natural fallback when `synthcontrol` has too few donors.

## Related kernels

- [`synthcontrol`](synthcontrol.md) : identification via donor pools
  instead of an instrument.
- [`rdd`](rdd.md) : identification via a threshold. RDD is often
  described as "fuzzy IV" in the sharp-vs-fuzzy taxonomy.

## References

- Wright, P. G. (1928). *The Tariff on Animal and Vegetable Oils*.
  Appendix B: the original IV derivation.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and
  Panel Data*, 2nd ed. Chapter 5 covers 2SLS variance derivation.
- Hayashi, F. (2000). *Econometrics*. Chapter 3.5 for the same result.
- Staiger, D., Stock, J. H. (1997). "Instrumental Variables Regression
  with Weak Instruments." *Econometrica* 65(3). The F < 10 rule.

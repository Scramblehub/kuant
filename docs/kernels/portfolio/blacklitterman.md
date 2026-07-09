# blacklitterman: Posterior mean and covariance

## Purpose

Combine an equilibrium (or CAPM) prior on expected returns with
investor views to produce a Bayesian posterior. Classical
mean-variance optimization on raw sample means is unstable: small
changes in `mu` produce large changes in weights. Black-Litterman
1990/1992 shrinks `mu` toward the prior by mixing it with the views
via their relative precisions.

Under Gaussian assumptions:

```math
Mprec  = (tau * Sigma)^{-1} + P' * Omega^{-1} * P
Mcov   = Mprec^{-1}
mu_bl  = Mcov * ((tau * Sigma)^{-1} * pi + P' * Omega^{-1} * Q)
Sigma_bl = Sigma + Mcov
w      = Sigma_bl^{-1} * mu_bl / lambda
```

`pi` and `mu_bl` are excess returns over the risk-free rate: subtract
rf from `prior_mean` before calling if working in raw returns.

## Public API

```python
from kuant.portfolio import blacklitterman

r = blacklitterman(
    prior_mean, prior_cov,
    P, Q,
    Omega=None,        # Idzorek proportional if None
    tau=0.05,
    risk_aversion=3.0,
)
r.posterior_mean       # (n,)
r.posterior_cov        # (n, n)
r.weights              # (n,) unconstrained mean-variance
r.views_shift          # posterior_mean - prior_mean
print(r.summary())
```

- `prior_mean` — (n,) equilibrium expected excess returns `pi`.
- `prior_cov` — (n, n) `Sigma`.
- `P` — (k, n) view-picker. Row `j` picks the assets involved in
  view `j`. Absolute views use a single 1 per row; relative views
  use `[..., +1, ..., -1, ...]`.
- `Q` — (k,) view expected returns.
- `Omega` — (k, k) view covariance. Diagonal entries are per-view
  uncertainty (larger = less confidence). If `None`, the Idzorek
  proportional method is used: `Omega = tau * P Sigma P'`, so view
  uncertainty tracks the prior uncertainty of the same linear
  combinations.
- `tau` — scalar shrinkage on the prior. Standard practice is
  `1 / T` where `T` is the number of prior observations; 0.05 is
  the Black-Litterman default.
- `risk_aversion` — `lambda` in the final weight step.

## Design decisions

### 1. Idzorek proportional Omega as default

If `Omega` is not passed we build `Omega = tau * P Sigma P'`,
following Idzorek 2005. This makes view uncertainty scale with the
prior uncertainty of the same linear combination, which is what a
non-expert user usually wants: a view on a low-volatility asset is
implicitly tighter than a view on a high-volatility asset.

A tiny ridge `1e-10 * I` is added before inversion so near-singular
`P Sigma P'` (redundant views) does not blow up.

### 2. Posterior formula in precision space

Working in precision (`Mprec`) instead of covariance keeps the
combining step numerically stable: adding two precisions is easier
than inverting-then-averaging two covariances. The single inversion
of `Mprec` is the only large solve.

### 3. Two matrix inversions, one solve

Three linear-algebra steps: `inv(tau * Sigma)`, `inv(Omega)`,
`inv(Mprec)`; then `linalg.solve(risk_aversion * Sigma_bl, mu_bl)`
for the weights. All at O(n^3) but a fresh factorization per call is
cheap for the sizes Black-Litterman is used at (typically 5 to 50
asset classes).

### 4. Excess-return convention

`posterior_mean` and `weights` are interpreted as EXCESS over the
risk-free rate. If you pass raw expected returns, subtract rf first.
This matches Black-Litterman 1992.

### 5. Unconstrained weights

The final `w = Sigma_bl^{-1} mu_bl / lambda` is the tangency
portfolio: not bounded, not normalized to sum to one. For a
long-only or fully-invested constraint, feed `posterior_mean` and
`posterior_cov` into a constrained solver (a QP, or
`kuant.portfolio.riskparity` if risk parity is acceptable).

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| `prior_cov` not square, or size mismatched to `prior_mean` | `KuantShapeError [KE-SHAPE-2D]` |
| `P` not 2D or `P.shape[1] != n` | `KuantShapeError [KE-SHAPE-2D]` |
| `Q.size != P.shape[0]` | `KuantShapeError [KE-SHAPE-EQUAL-LEN]` |
| `Omega.shape != (k, k)` | `KuantShapeError [KE-SHAPE-2D]` |
| `tau <= 0` or `risk_aversion <= 0` | `KuantValueError` from `require_positive` |

## Cross-check tests

- `test_view_shifts_posterior` — a positive view on asset 0 lifts
  posterior_mean[0] above prior mean.
- `test_weights_finite` — posterior solve stays finite.
- `test_shape_mismatch_rejected` — mismatched `P` columns.

`tests/portfolio/test_construction_batch6.py::TestBlackLitterman`.

## References

- Black & Litterman 1990, "Asset allocation: combining investor
  views with market equilibrium," Goldman Sachs Fixed Income
  Research.
- Black & Litterman 1992, "Global portfolio optimization," Financial
  Analysts Journal 48(5).
- Idzorek 2005, "A step-by-step guide to the Black-Litterman model."

## Related kernels

- `kuant.portfolio.riskparity` — allocation when views are absent.
- `kuant.portfolio.hrp` — robust allocation when `Sigma` is unstable
  or the user cannot supply meaningful views.

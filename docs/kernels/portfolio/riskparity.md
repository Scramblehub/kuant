# riskparity: Equal risk contribution portfolio

## Purpose

Find weights such that every asset (or every asset relative to a
supplied budget) contributes the same share of portfolio variance:

```math
w_i * (Sigma w)_i / (w' Sigma w) = b_i     for all i
```

`b = 1/n` by default (Maillard-Roncalli-Teiletche 2010 equal risk
contribution, ERC), or an explicit target budget summing to one.
Solved by cyclic coordinate descent on the closed-form per-coordinate
update.

## Public API

```python
from kuant.portfolio import riskparity

r = riskparity(cov)                              # ERC, 1/n budget
r = riskparity(cov, target=my_budget)            # custom risk budget
r = riskparity(cov, max_iters=1000, tol=1e-10)
r.weights                # (n,), sums to 1, strictly positive
r.risk_contributions     # (n,), sums to 1
r.portfolio_variance
r.converged, r.n_iters
print(r.summary())
```

- `cov` — 2D (n, n) covariance. Assumed positive-definite; degenerate
  diagonals (`Sigma[i,i] <= 0`) are skipped in the update.
- `target` — optional (n,) risk budget. Renormalized to sum to one.
  All entries must be strictly positive.
- `max_iters` — coordinate-descent cap. Default 500.
- `tol` — convergence tolerance on the largest per-coordinate weight
  change. Default `1e-8`.

## Design decisions

### 1. Cyclic coordinate descent, closed-form update

For coordinate `i`, holding the others fixed, ERC becomes a quadratic
in `w_i`:

```math
w_i^2 * Sigma_ii + w_i * partial_i - target_i = 0
partial_i = (Sigma w)_i - Sigma_ii * w_i
target_i  = b_i * (w' Sigma w)
```

The positive root is `w_i = (-partial + sqrt(disc)) / (2 Sigma_ii)`.
No Newton step, no line search. Cheap and robust.

### 2. Warm start at `1/n`

The uniform-weight initializer sits in the feasible region and is a
strong starting point for the ERC solution when `Sigma` is well
conditioned (the ERC weights are typically within a factor of 2 of
`1/n`). Coordinate descent then converges in tens of iterations
rather than hundreds.

### 3. Renormalize every outer iteration

`w = w / w.sum()` after each full sweep. Coordinate-descent updates
alone drift the sum; the renormalization keeps `w' Sigma w`
interpretable as portfolio variance throughout.

### 4. Custom risk budgets

Passing `target` implements "risk-budgeted parity": for instance,
`target = [0.5, 0.3, 0.1, 0.1]` says asset 0 should contribute
50% of portfolio risk. Budget vector is renormalized to sum to one
before use; strictly positive entries required (zero budget produces
zero weight, which the update cannot represent).

### 5. Convergence criterion

`max_change < tol` across a full coordinate sweep. Reports
`converged=False` and returns the best-effort weights if `max_iters`
is exhausted; the caller decides whether to trust the result or
retry with more iterations.

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| Non-square `cov` | `KuantShapeError [KE-SHAPE-2D]` |
| `target.size != n` | `KuantShapeError [KE-SHAPE-EQUAL-LEN]` |
| Any `target[i] <= 0` | `KuantValueError [KE-VAL-POSITIVE]` |
| `max_iters <= 0` or `tol <= 0` | `KuantValueError` from `require_positive` |
| `Sigma_ii <= 0` for some `i` | that coordinate skipped in the update |
| Non-convergence in `max_iters` | `converged=False`, best-effort `w` returned |

## Cross-check tests

- `test_equal_risk_contributions` — 6-asset ERC lands at
  `1/6 +- 1e-4` risk contribution each.
- `test_weights_sum_to_one`, `test_weights_positive`.
- `test_custom_target` — supplied budget produces monotone-matched
  contributions.
- `test_bad_shape_rejected`.

`tests/portfolio/test_construction_batch6.py::TestRiskParity`.

## References

- Maillard, Roncalli & Teiletche 2010, "The properties of equally
  weighted risk contribution portfolios," Journal of Portfolio
  Management 36(4).

## Related kernels

- `kuant.portfolio.hrp` — alternative covariance-only allocator,
  robust to singular `Sigma` (does not invert).
- `kuant.portfolio.blacklitterman` — when explicit views on
  expected returns exist and mean-variance is preferred.
- `kuant.portfolio.mintorsion` — diagnostic for the effective
  number of bets in the ERC solution.

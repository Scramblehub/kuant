# meancvar: Mean-CVaR portfolio LP

## Purpose

Solve the Rockafellar-Uryasev 2000 mean-CVaR problem:

```math
max_w  w' mu
s.t.   CVaR_alpha(w' R) <= gamma
       sum(w) = 1,  w >= 0   (long-only, optional)
```

Rockafellar-Uryasev showed that CVaR is linearly programmable by
introducing an auxiliary VaR variable `eta` and scenario slacks
`z_t`:

```math
CVaR_alpha = eta + (1 / ((1 - alpha) * T)) * sum_t max(0, -w' r_t - eta)
```

so the entire mean-CVaR problem reduces to a single LP in
`(w, eta, z)` of dimension `n + 1 + T`.

## Public API

```python
from kuant.portfolio import meancvar

# Minimize CVaR (no return target)
r = meancvar(returns, alpha=0.95)

# Maximize expected return subject to a CVaR cap
r = meancvar(returns, alpha=0.95, cvar_limit=0.05)

# Allow shorting
r = meancvar(returns, alpha=0.95, long_only=False)

r.weights, r.expected_return, r.cvar, r.var, r.status
print(r.summary())
```

- `returns` — 2D (T, n) historical scenarios. Requires `T >= 20`.
- `alpha` — confidence level in `(0.5, 0.999)`. Default 0.95.
- `cvar_limit` — upper bound on CVaR. If `None`, the objective
  becomes minimize-CVaR; if set, the objective becomes
  maximize-return with CVaR as a constraint.
- `long_only` — clip weights to `[0, inf)`. Default `True`.

Requires scipy for `scipy.optimize.linprog`. Raises
`KuantValueError [KE-DEP-MISSING]` if scipy is absent.

## Design decisions

### 1. Rockafellar-Uryasev LP reformulation

`max(0, -w' r_t - eta)` is not linear, but forcing an auxiliary
`z_t >= 0` with `z_t >= -w' r_t - eta` and pushing `z_t` down through
the objective produces the same optimum. The whole problem is then
linear in `(w, eta, z)`.

### 2. Two modes with one LP

- No `cvar_limit`: `c = [0, ..., 0, 1, 1/((1-alpha)T), ...]`. The
  objective is `eta + (1/((1-alpha)T)) sum z_t`, exactly the CVaR
  expression.
- With `cvar_limit`: objective flips to `c = [-mu, 0, 0, ..., 0]`
  and the CVaR expression migrates into the constraint set as
  `eta + (1/((1-alpha)T)) sum z_t <= cvar_limit`.

Same variable layout in both modes: dispatch is on the coefficient
vector `c` and the extra inequality row.

### 3. HiGHS solver

`scipy.optimize.linprog(method="highs")`. HiGHS is the default modern
LP solver in scipy and handles the `n + T + 1` variable scale of
mean-CVaR comfortably up to a few thousand scenarios.

### 4. Long-only via bounds, sum-to-one via equality

`bounds = [(0, None)] * n + [(None, None)] + [(0, None)] * T` for
long-only. `A_eq @ x = 1` on the `w` block. Both are cleaner than
extra inequality rows.

### 5. Solver failure is loud

If HiGHS returns `success=False`, `KuantValueError [KE-LP-FAILED]`
propagates with the solver message. This surfaces infeasible
combinations (`cvar_limit` too tight, degenerate scenario set)
early.

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| `returns.ndim != 2` | `KuantShapeError [KE-SHAPE-2D]` |
| `T < 20` scenarios | `KuantValueError [KE-VAL-MIN-CLEAN]` |
| `alpha` outside `(0.5, 0.999)` | `KuantValueError` from `require_range` |
| scipy missing | `KuantValueError [KE-DEP-MISSING]` |
| LP infeasible / unbounded | `KuantValueError [KE-LP-FAILED]` |

## Cross-check tests

- `test_weights_sum_to_one` — sum-to-one equality satisfied.
- `test_weights_nonneg_by_default` — long-only bound.
- `test_cvar_positive_for_random_returns` — CVaR loss magnitude
  non-negative on a symmetric-normal scenario set.
- `test_bad_alpha_rejected`, `test_too_few_scenarios_rejected`.

`tests/portfolio/test_construction_batch6.py::TestMeanCvar`.

## References

- Rockafellar & Uryasev 2000, "Optimization of conditional
  value-at-risk," Journal of Risk 2(3).

## Related kernels

- `kuant.portfolio.riskmetrics.ulcer_index` and
  `kuant.portfolio.drawdown` for downside-only summaries once
  `meancvar` produces a candidate `w`.
- `kuant.portfolio.hrp` for a comparison allocator that does not
  require an LP solve.

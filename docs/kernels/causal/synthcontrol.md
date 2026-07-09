# synthcontrol — Abadie-Diamond-Hainmueller synthetic control

## Purpose

Constructs a synthetic counterfactual for a single treated unit as a
convex combination of untreated donor units, weighted to match the
treated unit's pre-treatment outcome path. Post-treatment gap between
observed and synthetic is the treatment effect estimate.

Applies to policy analysis, natural experiments, and event studies
where random assignment is unavailable but the treated unit has many
plausible controls (states, firms, countries, tickers).

## Public API

```python
from kuant.causal import synthcontrol

res = synthcontrol(treated, donors, t_treat=25)
print(res.summary())
print(res.att, res.weights)
```

- `treated` : 1D array of length `T`. Outcome series for the treated
  unit.
- `donors` : 2D array of shape `(T, J)`. Outcome series for `J`
  candidate donor units.
- `t_treat` : int. First post-treatment index (0-indexed). Must satisfy
  `2 <= t_treat <= T - 1`.
- `n_iter` : int, default `2000`. Maximum projected-gradient steps.
- `tol` : float, default `1e-8`. Absolute convergence criterion on the
  objective.

Returns a `SynthControlResult` with fields `weights`,
`treated_pre_fit_rmse`, `treated_post`, `synthetic_post`, `gap_post`,
`att`, `n_donors`, `t_pre`, `t_post`.

## Design decisions

### 1. Simplex constraint by projected gradient descent

Weights are constrained to the probability simplex: `w_j >= 0`,
`sum(w) == 1`. The projection step uses the Duchi et al. 2008
sort-and-threshold routine (`_project_simplex`), applied after each
gradient step on the pre-period MSE.

Chosen over a QP solver because the projection is O(J log J) and needs
no external dependency. The pre-period MSE surface is convex on the
simplex, so gradient descent finds the global optimum given a small
enough step size.

### 2. Lipschitz-derived step size

Step size is `1 / ||X_pre_s||_2^2`. This is the reciprocal of the
Lipschitz constant of the gradient, guaranteeing monotone descent
without a line search. Scale-normalizing the pre-period series by
`std(y_pre)` keeps the step size well-behaved across problems.

### 3. ATT sign convention

`att = mean(gap_post) = mean(y_post - synthetic_post)`. Positive `att`
means the treated unit OUTPERFORMS its synthetic counterfactual after
treatment. Flip the sign at the call site if the natural direction of
your policy question is the opposite.

### 4. Pre-period cleaning is mandatory, post-period is masked

Rows in the pre-period with any non-finite value are dropped before
fitting. In the post-period, non-finite `gap_post` entries are excluded
from the `att` mean but preserved in `gap_post` for the caller.

Rejects if fewer than three clean pre-treatment observations survive
with `KE-VAL-MIN-CLEAN`. Three is the minimum for a non-degenerate
convex fit against multiple donors.

### 5. Convergence criterion is loss-delta, not gradient-norm

Loop terminates when `|prev_loss - loss| < tol`. Cheaper than
recomputing a projected-gradient norm each step, and the simplex
projection can leave residual gradient after convergence anyway.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `donors` not 2D | raises `KuantValueError` `[KE-SHAPE-2D]` |
| `len(treated) != T` (donors rows) | raises `[KE-SHAPE-EQUAL-LEN]` |
| `t_treat < 2` or `t_treat > T-1` | raises via `require_range` |
| Fewer than 3 clean pre-treatment rows | raises `[KE-VAL-MIN-CLEAN]` |
| All donors identical | weights are indeterminate but valid on the simplex; ATT still computed |
| Convergence before `n_iter` | early exit on `tol` |

## Cross-check tests

- `test_synthcontrol_recovers_att_and_weights` : synthesized DGP with
  `true_w = [0.6, 0.4, 0, 0, 0]` and a post-period bump of 3.0. The
  fitted `att` is within 0.3 of 3.0 and the top two weights carry
  > 0.85 mass. Simplex constraints hold to machine precision.
- `test_synthcontrol_shape_error` : 1D `donors` raises
  `KuantValueError`.
- `test_synthcontrol_len_mismatch` : rows/length mismatch raises.

## Direct usage in kuant

Event studies on portfolio holdings: treat one name as the "treated"
unit around an event, use sector peers as donors, and read the
post-event gap. Also composes with `kuant.qm.hmm` for regime-specific
counterfactuals (fit per state, not over the pooled sample).

## Related kernels

- [`iv`](iv.md) : alternative identification when a valid instrument
  exists but donors do not.
- [`rdd`](rdd.md) : threshold designs rather than one-unit-vs-many.

## References

- Abadie, A., Diamond, A., Hainmueller, J. (2010). "Synthetic Control
  Methods for Comparative Case Studies." *JASA* 105(490).
- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data
  Requirements, and Methodological Aspects." *JEL* 59(2).
- Duchi, J., Shalev-Shwartz, S., Singer, Y., Chandra, T. (2008).
  "Efficient Projections onto the L1-ball for Learning in High
  Dimensions." *ICML*.

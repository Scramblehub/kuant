# symbolicscan — Polynomial-symbolic regression scan

## Purpose

Build a polynomial expansion of your feature library (squares,
pairwise interactions, higher-order terms) and run LASSO with CV on
the expanded design matrix. Non-zero coefficients give you a
compact, interpretable polynomial equation.

Sits between `sindylasso` (linear-only) and `pinnscan` (fully
nonlinear black-box GBR): richer than pure linear, but still gives
you a readable formula.

## Public API

```python
from kuant.sindy import symbolicscan

result = symbolicscan(
    target, features,
    degree=2,             # default: squares + pairwise interactions
    alpha_grid=None,      # default: np.logspace(-5, -1, 30)
    n_splits=5,
    max_iter=10000,
    include_bias=False,
)
print(result.summary())
```

Returns `SymbolicScanResult` with `selected_terms` (dict of
polynomial-term name → coefficient), `alpha_selected`, `alpha_grid`,
`r2` (OOF), `n_terms_in_expansion`, `degree`, and `intercept`.

## Design decisions

### Uses sklearn's `PolynomialFeatures` for expansion

`PolynomialFeatures(degree=d, interaction_only=False)` gives:

- `degree=1`: identity (`x1`, `x2`, ...)
- `degree=2`: adds squares (`x1²`, `x2²`, ...) and pairwise
  interactions (`x1 x2`, `x1 x3`, ...)
- `degree=3`: adds cubes and triple interactions

Term names follow sklearn's naming: `'x1 x2'` for interaction,
`'x1^2'` for square. The `summary()` includes a rendered polynomial
equation.

### `KFold` with `shuffle=False`

Same time-series-safe CV as `sindylasso`.

### Auto-null diagnostic

If LASSO selects zero terms, the summary tags: "No compact symbolic
structure detected. Try pinnscan for a full nonlinear search, or
accept the null."

### Beware combinatorial blow-up

For `n_features` base features and degree `d`, the expansion has
`O(n^d)` columns. For `n_features = 20, degree = 3`, that's ~1770
terms — LASSO CV becomes slow. For research on rich libraries,
keep `degree ≤ 2` or drop obviously-irrelevant base features first.

## Value even when it returns null

A frequent outcome on daily-frequency financial data: a degree-2
polynomial expansion of a decent candidate library shows no
improvement over the pure linear scan. That negative result rules
out the compact-interaction hypothesis before proceeding to the fully
nonlinear pinnscan (more expensive, less interpretable).

## Related tools

- `kuant.sindy.sindylasso` — linear-only baseline
- `kuant.sindy.pinnscan` — fully nonlinear GBR baseline with permutation null

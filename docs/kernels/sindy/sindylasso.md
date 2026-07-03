# sindylasso — LASSO-with-CV feature-library scan

## Purpose

Given a target series and a rich library of candidate features (lags,
non-linear transforms, interactions, derivatives), fit L1-regularized
regression with cross-validated regularization strength. Sparse
selection + CV together answer "which of my features are actually
predictive, if any?"

## Public API

```python
from kuant.sindy import sindylasso

result = sindylasso(
    target,
    library={'x1': x1, 'x2': x2, 'x1_lag5': np.roll(x1, 5), ...},
    alpha_grid=None,      # default: np.logspace(-5, -1, 30)
    n_splits=5,
    max_iter=10000,
)
print(result.summary())
```

Returns `SindyLassoResult` with `selected_features` (dict of non-zero
coefficients), `alpha_selected`, `alpha_grid`, `r2` (OOF), and library
metadata.

## Design decisions

### KFold with `shuffle=False` — respects time-series ordering

CV folds are contiguous. Random shuffles would create leakage from
future to past in a time-series library.

### Auto-null diagnostic

The `summary()` method flags the classic null-signal signature: CV
picks alpha at the TOP of the search range AND the LASSO selects
zero features. This is the canonical residuals-after-multi-factor-wash
null pattern — an unambiguous "signal-to-noise is below threshold
across the entire library" verdict.

### NaN rows are dropped, not imputed

Rows where target OR any feature is NaN are removed before fitting.
Raises `ValueError` if fewer than 30 clean rows survive (too few to
CV meaningfully).

### OOF R² is recomputed at the chosen alpha

sklearn's `LassoCV.score` returns the training-set R². We manually
compute out-of-fold R² across the same CV splits so the reported
number matches the metric that drove alpha selection.

### Lazy sklearn dependency

`_require_sklearn()` at call time. Importing `kuant.sindy` doesn't
require sklearn.

## Canonical null pattern

The clearest null-signature this tool surfaces: CV picks the highest
alpha in the search range AND the LASSO selects zero non-zero
coefficients across an entire library of 20+ candidate features.
When that happens, the summary auto-diagnostic tag explicitly labels
it — you don't have to interpret the numbers, the tool tells you
signal-to-noise is below threshold across the whole library.

## Related tools

- `kuant.sindy.grangerscan` — parametric F-test alternative
- `kuant.sindy.permtest` — chain after sindylasso: shuffle target and
  re-scan to verify selection isn't a multiple-testing artifact
- `kuant.qm.belltest` — related but different: tests whether a JOINT
  model beats a classical AGGREGATE, not whether individual features
  survive selection

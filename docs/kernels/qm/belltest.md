# belltest — Bell-inequality-style aggregation test

## Purpose

Given a target and a set of features, ask: does a **joint model** carry
predictive information that classical aggregation cannot? The QM
Bell inequality tests whether a joint quantum state carries super-
classical correlations; this is the same idea applied to signal
aggregation.

Practical use: before committing engineering effort to a fancy joint
model (HMM, neural net, ensemble), verify it can actually beat the
best classical aggregate. If it can't, feature-level regime detection
has hit its theoretical ceiling — any remaining alpha must come from
picker-level or unobservable structure.

## Public API

```python
from kuant.qm import belltest

result = belltest(features, target, joint_model_fn=None, n_splits=5)
print(result.summary())
```

- `features` — dict[str, 1D np.ndarray] of named candidate features
- `target` — 1D np.ndarray
- `joint_model_fn` — optional `(X_train, y_train) -> y_pred_full` for
  a custom joint model. Default is a Gaussian Mixture posterior fed
  into a linear regressor (HMM-like).
- Returns `BellTestResult` with per-feature R², multi-linear OLS/Ridge/
  GradientBoosting R², joint model R², classical bound, and
  `joint_beats_bound` flag.

## Design decisions

### Classical bound = max R² across four classical predictors

- Per-feature linear regression (best single feature)
- Multi-feature OLS (linear joint)
- Multi-feature Ridge (regularized linear joint)
- GradientBoosting (nonlinear joint)

If your joint model can't beat all four, it's not doing anything a
classical predictor couldn't.

### Cross-validated R² throughout

K-fold with out-of-fold predictions; no in-sample overfit inflates
the classical bound. Standard 5-fold default.

### Lazy sklearn dependency

sklearn is imported at call time via `_require_sklearn()`. Importing
`kuant.qm` doesn't require sklearn; using this specific tool does.

## Real-world use in our research

Ran on V8's HMM joint posterior against 4 candidate features
(dβ_LV, breadth_5d, vix_level, hy_z5). Result: HMM R² = 0.0062,
classical bound R² = 0.0073 (from breadth_5d alone). HMM was BELOW
the classical bound by 15% — confirming the feature-level model was
just an efficient linear aggregator, not a source of new information.

This directed subsequent work away from feature-level engineering
and toward picker-level alpha.

## Related tools

- `kuant.qm.hmm` — the joint model this test was designed to evaluate
- `kuant.qm.zenoscan` — sibling QM-inspired experiment tool
- `kuant.sindy.permtest` — universal permutation p-value companion

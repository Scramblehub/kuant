# pinnscan — Nonlinear feature-library scan with permutation null

## Purpose

Given a target series and a feature library, fit a
GradientBoostingRegressor to get out-of-fold predictions, then run a
permutation test to confirm the fit is above the noise floor.

Where `sindylasso` searches for sparse LINEAR structure, `pinnscan`
searches for any NONLINEAR structure the library can express. If
LASSO came back empty but the library represents your physical
intuition, this is the next-most-common test to try.

## Public API

```python
from kuant.sindy import pinnscan

result = pinnscan(
    target, library,
    n_splits=5, n_perms=200,
    n_estimators=100, random_state=0,
)
print(result.summary())
```

Returns `PinnScanResult` with:
- `r2_oof` — out-of-fold R²
- `corr_oof` — out-of-fold correlation
- `feature_importances` — dict of GBR importances (single fit on clean data)
- `permutation_p` — p-value of `r2_oof` under target shuffling

## Design decisions

### Composes `permtest`

The nonlinear-fit null is computed by re-running the full OOF R² on
each shuffled target. Not the cheapest permutation loop, but the OOF R²
value drives the "is it real?" question directly, so it's the right
metric to permute.

### `KFold` with `shuffle=False`

Time-series ordering preserved. Shuffled CV folds would leak future
into past.

### Feature importances from a full fit

The `feature_importances` returned come from a SINGLE full-data GBR
fit. They're honest ranking indicators but not cross-validated. Use
them to decide which features to inspect first, not as evidence of
robustness.

### Auto-null diagnostic

If `permutation_p >= 0.05`, the summary tags: "OOF fit does NOT survive
permutation. The library carries no nonlinear signal above the noise
floor." Combined with a null `sindylasso` result, this typically means
the joint search space has been exhausted.

## Real-world use

V8 SINDy #8 (PINN-lite null). 20-feature library (dβ/dt × 8, factor
returns × 8, VIX, HY OAS, RSP breadth):

- OOF correlation: +0.0211
- OOF R²: 0.0004
- Quintile gate applied to OOF predictions: ΔCAGR +7.26pp, ΔCalmar +0.61
- **Permutation p = 0.50** (half of random-shuffle runs did as well)

This is the canonical positive-looking-but-actually-null pattern.
Without the permutation test we would have shipped a garbage signal.

## Related tools

- `kuant.sindy.sindylasso` — linear-only search on the same library
- `kuant.sindy.symbolicscan` — polynomial-symbolic middle ground
- `kuant.sindy.permtest` — the primitive this composes on

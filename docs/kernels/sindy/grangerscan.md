# grangerscan — Bonferroni-corrected Granger F-test scan

## Purpose

Given a target series and a library of `N` candidate predictors × `H`
horizons, run all `N · H` Granger causality F-tests and apply the
Bonferroni correction to the standard `α = 0.05`. Returns the list of
candidate/horizon pairs that pass the corrected threshold.

Cheap first-pass filter for "which of my macro/factor candidates could
in principle inform this target?"

## Public API

```python
from kuant.sindy import grangerscan

result = grangerscan(
    target,
    candidates={'vix': vix_series, 'rsp': rsp_series, ...},
    horizons=[1, 2, 5],
    alpha=0.05,
)
print(result.summary())
```

## Design decisions

### Bonferroni over the whole (candidate × horizon) grid

Corrected threshold = `α / (n_candidates * n_horizons)`. Controls
family-wise error rate at `α` even for large scans.

### statsmodels for the F-test (lazy dep)

`statsmodels.tsa.stattools.grangercausalitytests` implements the SSR-F
test. Imported at call time via `_require_statsmodels()`, so importing
`kuant.sindy` does not require statsmodels.

### Drops NaN rows automatically

Rows where either the target or the candidate is NaN are excluded
per-candidate. Skips candidates with fewer than 30 clean observations
(warns if `verbose=True`).

## Typical pattern

A common outcome when scanning a macro-factor library of a few dozen
candidates across a handful of horizons: several candidates pass the
Bonferroni threshold, but most of them are variants of the same
underlying macro variable (e.g. multiple `X_level`, `X_dret`,
`X_pct` all pass together). Post-filter for orthogonality against
your existing signal set — the "hits" that survive that filter are
often just one or two genuinely new candidates worth chasing.

## Interpretation warning — post-filter for orthogonality

Bonferroni-passing hits are frequently correlated with each other. If
you already have `X_1` as a shipped feature and `X_2 = f(X_1) + noise`,
both will pass. **Manually cross-check each hit against your existing
signal set before treating it as a new independent finding.**

## Related tools

- `kuant.sindy.permtest` — often chained after grangerscan to confirm
  hits with a shuffle-based null
- `kuant.qm.belltest` — companion "did we beat classical bounds" test

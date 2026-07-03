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

## Real-world use in our research

Ran on 35 macro factors × 3 horizons = 105 tests. Corrected threshold
= 0.05 / 105 = 0.00048. Five passed:

| Candidate | Horizon | F | p |
|---|---|---|---|
| VIX_level | 5d | 6.71 | 3e-6 |
| VIX_dret | 5d | 5.23 | 9e-5 |
| VIX_level | 2d | 4.86 | 2e-4 |
| VIX_level | 1d | 4.85 | 2e-4 |
| **RSP** | **5d** | **4.58** | **4e-4** |

Four of five were VIX-derivatives (redundant with V4's existing
mechanism). Only RSP-SPY breadth at 5d was genuinely orthogonal — and
it went on to inform a real production gate.

## Interpretation warning — post-filter for orthogonality

Bonferroni-passing hits are frequently correlated with each other. If
you already have `X_1` as a shipped feature and `X_2 = f(X_1) + noise`,
both will pass. **Manually cross-check each hit against your existing
signal set before treating it as a new independent finding.**

## Related tools

- `kuant.sindy.permtest` — often chained after grangerscan to confirm
  hits with a shuffle-based null
- `kuant.qm.belltest` — companion "did we beat classical bounds" test

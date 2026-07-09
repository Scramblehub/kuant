# correlations — Rank, nonlinear, and asymmetric correlation measures

## Purpose

Five workhorse alternatives to Pearson correlation (`rollcorr`), each
sensitive to a different class of dependence:

- `kendalltau` (Kendall 1938): tau-b rank correlation with tie
  adjustment. Robust to monotone transforms and outliers.
- `spearmanrank` (Spearman 1904): Pearson on ranks. Fast; also
  invariant to monotone transforms.
- `distancecorr` (Szekely-Rizzo 2007): zero if and only if `X` and
  `Y` are independent. Catches any (nonlinear) dependence.
  `O(n^2)` memory; capped at n = 2000 internally.
- `chatterjeexi` (Chatterjee 2020): rank-based coefficient bounded in
  `[-0.5, 1]`. Asymmetric in `(x, y)`; equals 1 when `Y` is a
  measurable function of `X`. `O(n log n)`.
- `downsidecorr`: Pearson conditional on BOTH series below a
  threshold. Tail co-movement / crash correlation.

## Public API

```python
from kuant.stats import (
    kendalltau, spearmanrank, distancecorr, chatterjeexi, downsidecorr,
)

r = kendalltau(x, y)
r = spearmanrank(x, y)
r = distancecorr(x, y)
r = chatterjeexi(x, y)          # asymmetric: chatterjeexi(x, y) != chatterjeexi(y, x)
r = downsidecorr(x, y, threshold=0.0)
```

- `x`, `y` — 1D arrays of equal length. Rows with any non-finite
  value in either series are dropped jointly.
- Returns `CorrelationResult(coef, p_value, n, method)`. `n` is the
  post-drop sample size (post-mask for downside).

## Design decisions

### 1. Shared input hygiene via `_check_pair`

Cast to float64, enforce 1D, enforce equal length, joint-mask
non-finite rows, then require at least 20 clean pairs:

```
KE-VAL-MIN-CLEAN: "only {n} paired finite values; need at least 20."
```

Length mismatch raises `KE-SHAPE-EQUAL-LEN`.

### 2. Kendall / Spearman: scipy first, reference fallback

Both prefer `scipy.stats.kendalltau` and `scipy.stats.spearmanr` for
tie handling and standard p-values. Fallbacks:

- Kendall: reference `O(n^2)` loop counting concordant, discordant,
  and tied pairs; tau-b formula with ties in both series; asymptotic
  z-test using variance `2(2n+5) / (9 n (n-1))`.
- Spearman: rank both series (`scipy.stats.rankdata`), take
  `corrcoef`, `t`-approximated two-sided p.

`_norm_sf` (a scipy-free normal survival function via `math.erf`)
covers the asymptotic p-values.

### 3. Distance correlation: capped at 2000 rows, deferred p-value

Distance correlation is `O(n^2)` in memory. Any input longer than 2000
uses the LAST 2000 rows only, matching kuant's tail-truncation
convention elsewhere.

Steps: pairwise absolute-distance matrices `a`, `b`; double-center
each; `dcov^2 = mean(A * B)`, `dvar_x = mean(A^2)`, `dvar_y = mean(B^2)`;
`dcor = sqrt(dcov^2 / sqrt(dvar_x * dvar_y))`.

`p_value = NaN` on purpose. Distance correlation's null distribution is
not distribution-free; permutation testing is deferred to
`kuant.nulltest.permtest`.

### 4. Chatterjee's xi: rank(y) sorted by x, then rank-jump sum

Sort `y` by `x`, rank the sorted `y`, and:

```
xi = 1 - 3 * sum(|diff(rank(y_sorted))|) / (n^2 - 1)
```

Asymmetric by construction: sorting is by `x`, so `xi(x, y) != xi(y, x)`
in general. Callers who need a symmetric summary should compute both
and take the mean (or use `distancecorr`).

Asymptotic null: `xi ~ N(0, 2/5 / n)`, so the reported two-sided p
uses `z = xi * sqrt(5 n / 2)`.

### 5. Downside correlation: joint threshold, min 20 downside pairs

Only rows where `x_i < threshold AND y_i < threshold` are included.
If fewer than 20 downside pairs remain, returns
`coef = NaN, p_value = NaN, n = n_down` rather than raising. This
lets callers fold `downsidecorr` into loops without wrapping in
try/except; a NaN downstream is the right signal.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` or `y.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| Unequal length | raises `KuantValueError` `KE-SHAPE-EQUAL-LEN` |
| Fewer than 20 clean pairs (all except `downsidecorr`) | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| `distancecorr` on `n > 2000` | silently truncated to last 2000 |
| `downsidecorr` with fewer than 20 downside pairs | returns `NaN` result |
| Perfect Pearson correlation in `downsidecorr` | p set to 0 (avoid divide by zero) |
| `distancecorr` on degenerate `dvar_x <= 0` | returns 0 coef, NaN p |
| Constant `y` in `chatterjeexi` | ranks tie; `xi` near 0 |

## Cross-check tests

- `test_perfect_positive_gives_one` (kendall) — `y = x` gives
  `coef > 0.99`
- `test_perfect_negative_gives_minus_one` (kendall) — `y = -x` gives
  `coef < -0.99`
- `test_monotone_transform_invariant` (spearman) — `y = exp(x)`:
  `coef > 0.99`
- `test_catches_nonlinear_dependence` (distance) — `y = x^2`:
  `coef > 0.3` (Pearson would be ~ 0)
- `test_functional_dependence_near_one` (chatterjee) — `y = sin(x)`
  on 1000 points: `xi > 0.7`
- `test_threshold_zero_default` (downside) — iid normals: `n_down`
  ~ 250 (roughly `n * 0.25`)
- `test_returns_nan_if_no_downside` (downside) — all-positive series:
  `NaN`

## References

- Kendall, M. G. (1938). "A new measure of rank correlation."
  Biometrika 30, 81-93.
- Spearman, C. (1904). "The proof and measurement of association
  between two things." American Journal of Psychology 15, 72-101.
- Szekely, G. J., Rizzo, M. L., Bakirov, N. K. (2007). "Measuring
  and testing dependence by correlation of distances." Annals of
  Statistics 35, 2769-2794.
- Chatterjee, S. (2021). "A new coefficient of correlation." JASA
  116, 2009-2022.

## Related

- `kuant.stats.rollcorr` — rolling Pearson baseline
- `kuant.stats.bdstest` — nonlinear dependence in a time-series
  context (uses embedding, not pairs)
- `kuant.nulltest.permtest` — permutation p-value for
  `distancecorr` and any other statistic

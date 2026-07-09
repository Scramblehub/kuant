# normalitytests — Normality tests for returns and residuals

## Purpose

Three normality tests, each with a distinct power profile:

- `jarquebera` (Jarque-Bera 1980): moment-based. Uses skewness and
  excess kurtosis. Fast, chi-square-2 null, reported alongside GARCH
  and OLS output.
- `andersondarling` (Anderson-Darling 1954): EDF-based, tail-weighted.
  More powerful than Kolmogorov-Smirnov in the tails, which is where
  return distributions differ from Gaussian.
- `shapirowilk` (Shapiro-Wilk 1965): order-statistic based. Highest
  power for small samples. Capped at n <= 5000 in scipy; kuant
  downsamples deterministically above that.

## Public API

```python
from kuant.stats import jarquebera, andersondarling, shapirowilk

jb = jarquebera(residuals)
ad = andersondarling(residuals)
sw = shapirowilk(residuals)
```

- `x` — 1D array. Non-finite values are dropped.
- Returns `NormalityResult(stat, p_value, n, test, extra)`. For
  Jarque-Bera, `extra` carries the sample skew and kurtosis (raw, not
  excess).

## Design decisions

### 1. Which test when

- `jarquebera` is the default for time-series work: fast, catches
  fat-tail deviations through kurtosis, and its statistic is
  interpretable on its own.
- `andersondarling` when the caller cares about the whole distribution,
  especially the tails; more powerful than JB against mixture-normal
  alternatives.
- `shapirowilk` when `n < 200` and the highest power is worth the
  scipy dependency. Above 5000, kuant thins the sample to 5000 via
  `np.linspace` indexing to stay inside scipy's supported range.

### 2. Jarque-Bera: population-moment normalization

Skew and kurt are computed on standardized data with `ddof = 0`:

```
mu = mean(x); sd = std(x, ddof=0)
z = (x - mu) / sd
JB = n / 6 * (skew(z)^2 + (kurt(z) - 3)^2 / 4)
```

This matches the JASA / econometrics convention (Kiefer-Salmon style).
If `sd < 1e-12` the test returns `stat = NaN`, `p = NaN` rather than
divide-by-zero.

### 3. Anderson-Darling: Stephens 1986 p-value bands

The A^2 statistic is corrected for finite `n`:

```
A^2_adj = A^2 * (1 + 0.75 / n + 2.25 / n^2)
```

Stephens' four-piece exponential-band approximation gives the p-value
over `A^2_adj` in `[0, inf)`. A cheap scipy-free path uses `math.erf`
elementwise for the Gaussian CDF; scipy's `norm.cdf` is preferred when
available.

### 4. Shapiro-Wilk: thin scipy wrapper with the 5000 cap

Scipy's `shapiro` is the reference; kuant's job is input validation
and deterministic downsampling above 5000 (via `np.linspace(0, n-1,
5000).astype(int)`) so the same input always yields the same result.

Missing scipy raises:

```
KE-DEP-MISSING: "requires scipy. -> Fix: pip install scipy"
```

Jarque-Bera and Anderson-Darling both work without scipy via the
`erf`-based fallbacks.

### 5. `n >= 20` floor

All three tests require at least 20 finite values:

```
KE-VAL-MIN-CLEAN: "only {n} finite values; need at least 20."
```

Below 20 the moment estimates for JB are unstable, the AD p-value bands
are outside their fit range, and Shapiro-Wilk loses power.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| Fewer than 20 finite values | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| Constant series (`sd ~ 0`) | JB/AD return `NaN` stat and p |
| `n > 5000` in `shapirowilk` | deterministic linspace downsample to 5000 |
| scipy missing (`shapirowilk`) | raises `KuantValueError` `KE-DEP-MISSING` |
| Ties in `andersondarling` | handled by scipy's CDF; kuant does not break them |

## Cross-check tests

- `test_jarquebera_gaussian_not_reject` — 1000-point normal: p > 0.05
- `test_jarquebera_uniform_reject` — uniform (kurt = 1.8): p < 0.05
- `test_andersondarling_gaussian_p_high` — 500-point normal: p > 0.05
- `test_shapirowilk_gaussian_p_high` — 200-point normal: p > 0.05

## References

- Jarque, C. M., Bera, A. K. (1980). "Efficient tests for normality,
  homoscedasticity and serial independence of regression residuals."
  Economics Letters 6, 255-259.
- Anderson, T. W., Darling, D. A. (1954). "A test of goodness of fit."
  JASA 49, 765-769.
- Stephens, M. A. (1986). "Tests based on EDF statistics." In
  D'Agostino & Stephens (eds.), Goodness-of-Fit Techniques, Marcel
  Dekker.
- Shapiro, S. S., Wilk, M. B. (1965). "An analysis of variance test
  for normality (complete samples)." Biometrika 52, 591-611.

## Related

- `kuant.stats.autocorrtests.ljungbox` — pair with normality tests for
  residual diagnostics
- `kuant.stats.correlations.chatterjeexi` — dependence beyond linear /
  Gaussian
- `kuant.stats.spectralentropy` — complementary complexity metric

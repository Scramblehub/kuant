# stationarity: Unit-root and stationarity tests

## Purpose

Four canonical stationarity / unit-root tests behind a single
`StationarityResult` dataclass:

- `adftest`: Augmented Dickey-Fuller. Null: series has a unit root.
- `kpsstest`: KPSS. Null: series is trend-stationary (INVERTED).
- `phillipsperrontest`: Phillips-Perron. Null: unit root; robust
  to serial correlation.
- `varianceratiotest`: Lo-MacKinlay. Null: random walk (IID returns).

Same input contract for all four: a 1D array with at least 20 finite
observations, an optional significance level, and an `is_stationary`
boolean derived from the p-value under the test's own null convention.

## Public API

```python
from kuant.stats import adftest, kpsstest, phillipsperrontest, varianceratiotest

adf = adftest(series, alpha=0.05, regression='c')
kps = kpsstest(series, alpha=0.05, regression='c')
pp  = phillipsperrontest(series, alpha=0.05, regression='c')
vr  = varianceratiotest(series, lags=2, alpha=0.05)
```

- `series`: 1D array. NaNs are dropped; need >= 20 finite.
- `alpha`: significance level in `(0, 1)`.
- `regression`:
  - `adftest`: `'c'`, `'ct'`, `'ctt'`, `'n'`.
  - `kpsstest`: `'c'`, `'ct'`.
  - `phillipsperrontest`: `'c'`, `'ct'`, `'n'`.
- `lags`: variance-ratio aggregation horizon (`k` in
  `Var(k-period) / (k * Var(1-period))`).

## Optional dependencies

- `adftest`, `kpsstest` rely on `statsmodels.tsa.stattools`.
- `phillipsperrontest`, `varianceratiotest` rely on the optional
  `arch` package's `unitroot.PhillipsPerron` and `unitroot.VarianceRatio`
  implementations. A missing install raises `KuantDependencyError`
  with an install hint.

## Design decisions

### 1. One `StationarityResult` dataclass across all four tests

Callers should be able to swap between tests without changing
downstream code. Every test returns:

```python
StationarityResult(
    statistic,
    p_value,
    is_stationary,
    test,
    null_hypothesis,
    n,
)
```

`test` names the kernel that produced the result; `null_hypothesis`
carries a human-readable statement of what a small p rejects.

### 2. Null conventions are INVERTED between ADF and KPSS

This is the most common source of confusion. kuant handles the flip
inside `is_stationary` so users do not have to remember:

- **ADF**: null = unit root. Small p REJECTS unit root, so
  `is_stationary = (p < alpha)`.
- **KPSS**: null = trend-stationary. Small p REJECTS stationarity,
  so `is_stationary = (p >= alpha)`.

`phillipsperrontest` follows the ADF convention. `varianceratiotest`
uses `is_stationary = (p < alpha)` where `True` means "reject the
random-walk null" (mean-reversion or momentum detected).

### 3. Minimum sample guard: 20 finite points

Every test has poor small-sample properties. The 20-obs floor is a
sanity check, not a rigorous power calculation. Callers with smaller
samples get a `KuantValueError` explaining the shortfall rather than
a silently unreliable p-value.

### 4. ADF autolag: AIC

`adfuller(..., autolag='AIC')` selects the lag order by minimizing
AIC. This is the statsmodels default and the most common choice in
the finance literature. Fixed-lag callers can wrap statsmodels
directly if they need a different criterion.

### 5. KPSS `nlags='auto'` and warning suppression

`kpss` emits an `InterpolationWarning` whenever the requested p-value
falls outside the tabulated grid. This is expected and not actionable
for the caller, so we suppress it inside the kernel.

### 6. Phillips-Perron via optional dependency

`statsmodels` does not ship a Phillips-Perron implementation, so
kuant delegates to the optional dependency named in the section
above. If the import fails, `require_dep` raises with an install
hint. The `regression` argument maps `'c'`, `'ct'`, `'n'` onto the
underlying implementation's `trend` argument one-for-one.

### 7. Variance-ratio interpretation

The variance-ratio statistic at lag `k` is

```math
VR(k) = Var(k-period returns) / (k * Var(1-period returns))
```

Under a random walk, `VR(k) = 1` for all `k`. `VR > 1` indicates
positive serial correlation (momentum); `VR < 1` indicates negative
serial correlation (mean reversion). The p-value tests whether the
observed `VR(k)` is significantly different from 1.

The input should be LEVELS (prices or log-prices); the internal
differencing is handled by the underlying implementation.

## Return shape

**StationarityResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `statistic` | float | Raw test statistic |
| `p_value` | float | Approximate p under the test's null |
| `is_stationary` | bool | True if the series is judged stationary |
| `test` | str | Kernel name |
| `null_hypothesis` | str | Human-readable null |
| `n` | int | Finite observations used |

`.summary()` returns a formatted multi-line string.

## Examples

```python
>>> import numpy as np
>>> from kuant.stats import adftest, kpsstest
>>> rng = np.random.default_rng(0)
>>> # White noise is stationary.
>>> x = rng.standard_normal(500)
>>> adftest(x).is_stationary
True
>>> kpsstest(x).is_stationary
True
>>> # Random walk is NOT stationary.
>>> rw = np.cumsum(rng.standard_normal(500))
>>> adftest(rw).is_stationary
False
```

## Related kernels

- `kuant.stats.hurstrs`, `kuant.stats.rollhurst`: Hurst exponent as
  a continuous mean-reversion / momentum measure.
- `kuant.stats.dfa`: Detrended fluctuation analysis for the same
  question in a non-stationary friendly form.
- `kuant.nulltest.bootstrap_ic`: block-bootstrap under serial
  correlation when parametric assumptions are suspect.

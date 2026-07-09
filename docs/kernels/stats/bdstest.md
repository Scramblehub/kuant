# bdstest — Brock-Dechert-Scheinkman test for iid

## Purpose

Nonlinear iid test based on the correlation integral across embedding
dimensions (Brock, Dechert, Scheinkman, LeBaron 1996). Detects
nonlinear structure that linear autocorrelation tests (Ljung-Box,
Durbin-Watson) miss.

Typical use: run OLS or ARMA, whiten the residuals of linear
autocorrelation, then apply `bdstest`. Rejection means the residuals
still carry nonlinear dependence (regime, GARCH, chaos, hidden state).

Under H0 (iid), the statistic is asymptotically `N(0, 1)`.

## Public API

```python
from kuant.stats import bdstest

r = bdstest(x, m=2, epsilon=None)
```

- `x` — 1D array. Non-finite dropped. Requires `n >= 100`.
- `m` — embedding dimension. Integer in `[2, 10]`.
- `epsilon` — sup-norm distance threshold. Default `0.7 * std(x, ddof=1)`
  per Kanzler (1999). Must be positive.
- Returns `BdsResult(stat, p_value, m, epsilon, n)`.

## Design decisions

### 1. Correlation integral by sup-norm neighbor counting

`_correlation_integral(x, m, eps)` embeds `x` into `m` dimensions
(sliding window), then counts pairs `i < j` with
`max_k |x_{i+k} - x_{j+k}| <= eps` and normalizes by `n(n-1)/2`.

Sup-norm is standard BDS: it makes the joint-neighbor probability
factor cleanly under H0, so `C_m(eps)` approaches `C_1(eps)^m` for
iid `x`.

### 2. Choice of `m` and `epsilon`

- `m`: 2 or 3 is the reported standard. Higher `m` needs more data
  because the number of neighbors shrinks as `C_1^m`.
- `epsilon`: `0.7 * sd` is the Kanzler (1999) rule; it lands in the
  middle of the correlation integral's usable range (avoids the
  no-neighbor and all-neighbor limits).

### 3. Simplified variance approximation

kuant uses a simplified form of Brock et al 1996 equation 2.5 that
omits higher-order moment corrections:

```
var = 4 * ( K^m + 2 sum_{j=1}^{m-1} K^(m-j) C_1^(2j)
            + (m-1)^2 C_1^(2m) - m^2 K C_1^(2m-2) )
```

where `K = C_1(2 * eps)`. Consequences (documented in the docstring):

- `m = 2` recovers the exact form.
- `m >= 3` gives a CONSERVATIVE z-statistic (understates
  significance).

For rigorous inference at higher `m`, wrap in `kuant.nulltest`
bootstrap. Deliberate: correctness of the sign of the reported effect
matters more than a tight tail p-value, and the boot path is cheap.

### 4. NaN guards on degenerate correlation integrals

Any of `C_1 <= 0`, `C_m <= 0`, or `var <= 0` returns
`stat = NaN, p = NaN`. Happens on constant series or when `eps` is
too small for the sample; no crash, downstream sees NaN.

### 5. Input floors

```
KE-VAL-MIN-CLEAN: "only {n} finite values; need at least 100."
KE-VAL-RANGE:     raised via require_range if m outside [2, 10]
KE-VAL-POSITIVE:  "'epsilon' must be positive, got {epsilon}."
```

`m = 1` is rejected because the test needs at least two embedding
dimensions to compare.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| `n < 100` | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| `m` outside `[2, 10]` | raises `KuantValueError` `KE-VAL-RANGE` |
| `epsilon <= 0` | raises `KuantValueError` `KE-VAL-POSITIVE` |
| Constant series | `C_1 = 1`, `var = 0`; returns NaN |
| `m >= 3` | z-statistic conservative (bias toward null) |
| Logistic-map deterministic input | z well above the noise level |

## Cross-check tests

- `test_iid_stat_near_zero` — 800-point Gaussian noise: `|stat| < 3`
- `test_logistic_map_stat_elevated` — deterministic `r = 4` logistic
  map, `m = 3`: `|stat| > 1.5` (threshold set above the noise floor
  given the conservative variance)
- `test_bad_epsilon_rejected` — `epsilon = -0.1` raises

## References

- Brock, W. A., Dechert, W. D., Scheinkman, J. A., LeBaron, B. (1996).
  "A test for independence based on the correlation dimension."
  Econometric Reviews 15, 197-235.
- Kanzler, L. (1999). "Very fast and correctly sized estimation of
  the BDS statistic." Working paper, Oxford.

## Related

- `kuant.stats.autocorrtests.ljungbox` — linear autocorrelation test;
  pair with BDS on residuals
- `kuant.stats.correlations.distancecorr`,
  `kuant.stats.correlations.chatterjeexi` — dependence in the pair
  setting (not time series)
- `kuant.nulltest` — bootstrap null for the m >= 3 case

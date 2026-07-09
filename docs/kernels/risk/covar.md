# covar — Adrian-Brunnermeier CoVaR via quantile regression

## Purpose

CoVaR measures the VaR of asset `X` conditional on asset `Y` being
at its own VaR. In the systemic-risk framing of Adrian-Brunnermeier
2016, `X` is "the system" and `Y` is an individual institution, and
CoVaR reads as "how much does the system lose when this institution
is in tail distress." The framing works either direction; the kernel
is agnostic.

Delta-CoVaR compares CoVaR (`Y` at its VaR) to the median-conditional
counterpart (`Y` at its median), isolating the incremental tail
exposure that the tail dependence itself creates.

## Public API

```python
from kuant.risk import covar

result = covar(returns_x, returns_y, alpha=0.95)
print(result.summary())
print(result.covar, result.delta_covar)
```

- `returns_x`, `returns_y`: 1D arrays of equal length. Non-finite
  pairs stripped jointly.
- `alpha`: confidence level in `[0.5, 0.9999]`. Default `0.95`.

Returns `CoVarResult` with fields `covar`, `delta_covar`,
`var_x_uncond`, `var_y_uncond`, `q_regression_slope`, `alpha`, `n`.

## Design decisions

### 1. Quantile regression on losses

Fit `loss_X = a + b * loss_Y` under the tau=alpha check loss:

```math
\min_{a, b} \sum_t \rho_\alpha(\text{loss}_X^t - a - b \cdot \text{loss}_Y^t)
```

with `rho_alpha(u) = u * (alpha - 1(u < 0))`. The fitted line
evaluated at `loss_Y = VaR_Y` gives CoVaR; evaluated at
`loss_Y = median(loss_Y)` gives the CoVaR-median. Delta-CoVaR is the
difference. Loss-space is a sign flip from returns so downstream
outputs stay positive.

### 2. Subgradient descent on standardized inputs

`_quantile_regression_1d` standardizes both series to zero mean and
unit stdev, runs subgradient descent on `(a, b)` for `n_iter = 2000`
steps with a decaying learning rate, then undoes the standardization
in closed form. Standardization keeps a single learning rate valid
across problem scales.

The alternative is `scipy.optimize` (interior point) or `statsmodels`
`QuantReg`. Subgradient descent avoids a hard dependency and is
accurate enough for the diagnostic purpose here (test suite hits it
with `slope > 0.3` and `< 0.25` gates on synthetic data). For
publication-grade CoVaR estimation, swap in `statsmodels`.

### 3. Positive-loss sign convention

CoVaR and delta-CoVaR are positive loss magnitudes. A positive
`q_regression_slope` means larger `Y` losses predict larger `X`
losses, that is, the two are tail-linked.

### 4. Delta-CoVaR uses median as the null

Follows the Adrian-Brunnermeier 2016 convention: contrast CoVaR at
the tail against CoVaR at the median of `Y`. Isolates the tail
part of the dependence rather than mixing in unconditional level.

### 5. Minimum sample and length parity

- `n_finite_pairs < 100` raises `KE-VAL-MIN-CLEAN`.
- `len(returns_x) != len(returns_y)` raises `KE-SHAPE-EQUAL-LEN`
  before any NaN masking.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `returns_x` and `returns_y` different lengths | `KuantValueError` with `KE-SHAPE-EQUAL-LEN` |
| Fewer than 100 paired finite values | `KuantValueError` with `KE-VAL-MIN-CLEAN` |
| `alpha` out of range | `KuantValueError` with `KE-VAL-RANGE` |
| Independent series | `q_regression_slope ~ 0`, `delta_covar ~ 0` |
| Perfectly correlated series | slope near 1, CoVaR near unconditional VaR_X |
| Non-finite entries in either input | paired stripping via `isfinite AND isfinite` |
| `sigma_x = 0` or `sigma_y = 0` inside regression | `sx = 1.0` or `sy = 1.0` fallback prevents divide-by-zero |

## Cross-check tests

- `test_covar_positive_dependence_gives_positive_slope`: 2k draws,
  `asset = 0.7 * sys + 0.3 * noise`. Recovered slope `> 0.3`,
  `delta_covar > 0`.
- `test_covar_independent_gives_low_delta`: 2k independent Gaussians.
  `|slope| < 0.25`, `|delta_covar| < 0.005`.
- `test_covar_length_mismatch`: `len(x) = 200`, `len(y) = 300`
  raises.
- `test_covar_min_clean_gate`: only 2 finite pairs raises with
  `KE-VAL-MIN-CLEAN`.

## Direct usage in kuant

Systemic-risk diagnostic on a paired series. Run over a rolling
window to track how tail-linked a book is to a benchmark under
different regimes. Delta-CoVaR trending up is a live warning that
tail dependence is intensifying.

## Related kernels

- `kuant.risk.mes`: complementary systemic measure. CoVaR conditions
  on a quantile POINT of the system; MES averages across a tail
  SLAB. Use both when the framing allows.
- `kuant.risk.cornishfishervar`, `kuant.risk.evtvar`: univariate
  tail estimators for the individual legs.

## References

- Adrian, T., Brunnermeier, M. 2016. "CoVaR." *American Economic
  Review* 106(7): 1705-1741.
- Koenker, R., Bassett, G. 1978. "Regression quantiles."
  *Econometrica* 46(1): 33-50.

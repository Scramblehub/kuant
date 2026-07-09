# cornishfishervar — Cornish-Fisher expansion VaR

## Purpose

Value-at-Risk that adjusts the Gaussian quantile for the sample's
skew and excess kurtosis via the Cornish-Fisher (1937) series. When
returns depart mildly from Gaussian, this yields a materially
different VaR at near-zero incremental cost over the naive formula.

Sits between plain Gaussian VaR (blind to fat tails) and `evtvar`
(fits the tail explicitly). Use it when the series is roughly
symmetric with moderate excess kurtosis; escalate to `evtvar` past
`|skew| > 1` or `|excess kurt| > 7`.

## Public API

```python
from kuant.risk import cornishfishervar

result = cornishfishervar(returns, alpha=0.95)
print(result.summary())
print(result.var)  # POSITIVE loss magnitude
```

- `returns`: 1D array-like. Non-finite entries are stripped.
- `alpha`: confidence level in `[0.5, 0.999]`. Default `0.95`.

Returns `CornishFisherVarResult` with fields `var`, `z_gaussian`,
`z_cf`, `mean`, `std`, `skew`, `excess_kurtosis`, `alpha`, `n`.

## Design decisions

### 1. Cornish-Fisher expansion, third and fourth cumulants

```math
z_{cf} = z + (z^2 - 1) \frac{S}{6}
           + (z^3 - 3z) \frac{K}{24}
           - (2 z^3 - 5z) \frac{S^2}{36}
```

with `z = Phi^{-1}(1 - alpha)`, `S` the sample skew, and `K` the
sample excess kurtosis. VaR is then `-(mu + z_cf * sigma)`, reported
as a positive loss.

Third + fourth-cumulant terms only. Higher-order terms carry high
variance and rarely improve fit for the sample sizes we care about.

### 2. Warns when the expansion breaks down

Cornish-Fisher is a truncated Edgeworth series. It becomes
non-monotone in the tail once skew or kurtosis leaves a bounded
region (roughly `|S| <= 1`, `|K| <= 7` for standard use). Past that,
the reported VaR can be smaller than the Gaussian VaR at a higher
confidence level. A `KuantNumericWarning` with code
`KW-CF-EXPANSION-INVALID` fires whenever either bound is breached,
and the message points at `evtvar` as the correct escalation.

### 3. Scipy optional, rational fallback

`_norm_ppf` prefers `scipy.stats.norm.ppf`. When scipy is not
installed, a Beasley-Springer-Moro rational approximation gives
absolute error below `1e-9` across the `alpha` range we permit. No
hard scipy dependency.

### 4. Positive-loss sign convention

VaR is `-(mu + z_cf * sigma)`. Positive means "you lose this much."
A constant series with `mu > 0` would report a NEGATIVE loss under
the naive formula; the kernel clamps to `max(-mu, 0)` in that
degenerate branch so callers never see a negative VaR.

### 5. Minimum sample

`n_finite < 30` raises `KE-VAL-MIN-CLEAN`. Below that, the fourth
sample moment is too noisy for the expansion to make sense.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `sigma < 1e-15` (constant series) | returns `var = max(-mu, 0)`, moments as NaN |
| `alpha` outside `[0.5, 0.999]` | `KuantValueError` with `KE-VAL-RANGE` |
| Fewer than 30 finite values | `KuantValueError` with `KE-VAL-MIN-CLEAN` |
| `|skew| > 1` or `|excess kurt| > 7` | `KuantNumericWarning` with `KW-CF-EXPANSION-INVALID`, result still returned |
| Non-finite entries in input | stripped before fitting |
| Non-1D input | `KuantValueError` |

## Cross-check tests

- `test_cornishfishervar_gaussian_matches_gaussian_var`: 50k
  standard-normal draws. CF quantile within `0.05` of Gaussian, VaR
  within `0.001` of `1.6449 * sigma`.
- `test_cornishfishervar_fat_tail_var_larger_than_gaussian`: 20k
  Student-t (df=3) draws. CF VaR strictly above the Gaussian
  `2.326 * sigma - mu` at `alpha = 0.99`.
- `test_cornishfishervar_warns_on_extreme_kurtosis`: single 50-sigma
  spike inserted into an otherwise Gaussian series triggers
  `KW-CF-EXPANSION-INVALID`.
- `test_cornishfishervar_rejects_short_series`: `n = 10` raises.

## Direct usage in kuant

Cheap first pass for VaR reporting on any near-Gaussian return
stream. When the warning fires, the calling code should escalate to
`evtvar` and record which regime it fell back to.

## Related kernels

- `kuant.risk.evtvar`: the correct estimator once
  `KW-CF-EXPANSION-INVALID` fires.
- `kuant.risk.esbootstrap`: gives a CI band on ES; pair with the
  point VaR from here for a fuller tail-risk picture.

## References

- Cornish, E., Fisher, R. 1937. "Moments and cumulants in the
  specification of distributions." *Revue de l'Institut International
  de Statistique*.
- McNeil, A., Frey, R., Embrechts, P. 2015. *Quantitative Risk
  Management*, 2nd ed., ch. 2 on quantile-based measures.

# wavelethurst — Abry-Veitch wavelet Hurst estimator

## Purpose

Wavelet-based Hurst estimator following Abry-Veitch (1998). For a
self-similar process with Hurst `H`, the variance of discrete wavelet
detail coefficients at dyadic scale `j` scales as

```
var_j ~ 2 ^ (j * (2H - 1))
```

so a straight-line fit of `log2(var_j)` against `j` yields
`H = (slope + 1) / 2`.

Uses a Haar wavelet decomposition to avoid a PyWavelets dependency.
Haar is sufficient for Hurst inference; smoother wavelets improve the
constant factor but not the asymptotic scaling.

## Public API

```python
from kuant.stats import wavelethurst

r = wavelethurst(x, scale_lo=2, scale_hi=7)
```

- `x` — 1D array. Non-finite values dropped. Requires `n >= 128`.
- `scale_lo`, `scale_hi` — dyadic scales included in the log-log fit.
  Default `[2, 7]` covers 4- to 128-sample resolutions.
- Returns `WaveletHurstResult(hurst, scales, log_var, intercept)`.

## Design decisions

### 1. Non-redundant Haar DWT

`_haar_dwt_coefs_by_scale` iterates: truncate to even length, split
into pairs, form `(c1 - c2) / sqrt(2)` details and `(c1 + c2) / sqrt(2)`
approximations, keep the details, recurse on the approximation. Cost
`O(n)` across all scales; no external DWT library.

Halving at each level means only `floor(log2(n))` scales are
representable; the code caps `n_scales` at that ceiling.

### 2. Log-log fit on variance of details

At each scale `j`, `var_j = var(coefs[j], ddof=1)` (sample variance).
Fit is `polyfit(fit_scales, log2(var), 1)`; slope `s` gives
`H = (s + 1) / 2`.

Restricting the fit to `[scale_lo, scale_hi]` drops the finest scale
(dominated by measurement noise) and the coarsest few (poor sample
size), leaving the linear scaling regime.

### 3. Scale range validation

`scale_lo in [1, 15]`; `scale_hi in [scale_lo + 1, 20]`. Both go
through `require_range` and raise `KE-VAL-RANGE` on violation. If
`n_scales <= scale_lo` after capping to `floor(log2(n))`:

```
KE-VAL-RANGE: "with n={n}, max scale is {max_possible_scales};
scale_lo ({scale_lo}) leaves no fit range."
```

### 4. `n >= 128` floor

Below 128 the dyadic decomposition has too few scales to fit a
straight line:

```
KE-VAL-MIN-CLEAN: "only {n} finite values; need at least 128 for a
stable dyadic decomposition."
```

### 5. Requires at least 3 valid scales in the fit range

After masking non-finite / zero variances:

```
KE-VAL-MIN-CLEAN: "fewer than 3 valid scales in the fit range."
```

Two-point fits give a slope but zero uncertainty diagnostics; three is
the minimum where the linear regime is even checkable.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| `n < 128` | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| `scale_lo` or `scale_hi` outside allowed range | raises `KuantValueError` `KE-VAL-RANGE` |
| `scale_lo >= floor(log2(n))` | raises `KuantValueError` `KE-VAL-RANGE` |
| `< 3` valid scales in fit range | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| Odd-length intermediate signal | last sample dropped before halving |
| Constant series | zero-variance scales masked; likely raises min-clean |

## Cross-check tests

- `test_noise_hurst_near_half` — 4096-point Gaussian noise:
  `H in (0.35, 0.65)` (theoretical 0.5)
- `test_too_short_rejected` — 50-point input raises `KuantValueError`
- `test_summary` — result renders

## References

- Abry, P., Veitch, D. (1998). "Wavelet analysis of long-range-
  dependent traffic." IEEE Transactions on Information Theory 44,
  2-15.
- Haar, A. (1910). "Zur Theorie der orthogonalen Funktionensysteme."
  Mathematische Annalen 69, 331-371.

## Related

- `kuant.stats.higuchihurst` — fractal-dimension route; use as second
  opinion
- `kuant.stats.localwhittle` — semiparametric long-memory `d`
- `kuant.stats.hurstrs` — classical R/S
- `kuant.stats.dfa`, `kuant.stats.mfdfa` — DFA family

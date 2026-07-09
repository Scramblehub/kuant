# localwhittle — Robinson local Whittle long-memory estimator

## Purpose

Semiparametric estimator of the long-memory parameter `d` (Robinson
1995) from the low-frequency portion of the periodogram. Reports
`d in (-0.5, 1)` and the derived Hurst `H = d + 0.5` (for stationary
long-memory series).

Local Whittle likelihood:

```
L(d) = log( mean(w_j^(2d) * I_j) ) - (2d / m) * sum(log w_j)
```

Minimizing over `d` uses only the first `m` Fourier frequencies, which
gives sharp asymptotic efficiency for pure long-memory processes and
insulates the estimate from short-memory contamination further out on
the spectrum.

Compared to R/S and DFA:

- Sharper asymptotic efficiency for pure long-memory.
- Less sensitive to short-memory contamination.
- Well-defined asymptotic SE under Gaussian assumptions: `0.5 / sqrt(m)`.

## Public API

```python
from kuant.stats import localwhittle

r = localwhittle(x, m=None)
```

- `x` — 1D array. Non-finite dropped. Requires `n >= 200`.
- `m` — number of Fourier frequencies to fit. Default
  `round(n ** 0.7)` per Robinson's optimal bias-variance rate.
- Returns `LocalWhittleResult(d, hurst, m, n, se)`.

## Design decisions

### 1. Periodogram on the positive-half frequencies

Demean, take `np.fft.fft`, form the periodogram
`I_j = |X_j|^2 / (2 pi n)`, keep indices `1..n//2`. Fourier
frequencies are `w_j = 2 pi j / n`.

### 2. Robinson `m = n ** 0.7` default

Under Robinson's bias-variance tradeoff, `m = n ** 0.7` balances the
bias from higher-frequency short-memory contamination against the
variance from a small fit sample. `m` is validated:

```
require_range(m, "m", kernel="localwhittle", lo=10, hi=n // 2)
```

which raises `KE-VAL-RANGE` if out of bounds.

### 3. Grid scan then golden-section refinement over `d`

The objective `neg_ll(d)` is smooth but the closed-form solution is
implicit. kuant does:

1. Coarse grid: `linspace(-0.49, 0.99, 149)`
2. Bracket the argmin with `[grid[best - 2], grid[best + 2]]`
3. 40 iterations of golden-section (phi = (sqrt(5) - 1) / 2)

The bracket keeps refinement inside the stationarity range
`d in (-0.5, 1)`; 40 iterations reduce the interval by
`phi^40 ~ 2e-9`.

### 4. `G(d) <= 0` returns `+inf` in the objective

If the geometric term underflows to zero on a specific `d`, the
objective returns `+inf` so the optimizer moves away from it. Guards
against NaN propagation on pathological inputs.

### 5. Asymptotic SE is `0.5 / sqrt(m_eff)`

Robinson (1995) shows `sqrt(m) (d_hat - d) -> N(0, 1/4)` under
regularity, so `SE(d_hat) = 0.5 / sqrt(m)`. `m_eff` is `min(m, |I_pos|)`
in case `m` exceeded the number of available frequencies.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| `n < 200` | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| `m` outside `[10, n // 2]` | raises `KuantValueError` `KE-VAL-RANGE` |
| `G(d) = 0` on grid | that `d` gets `+inf`; optimizer moves on |
| Constant series | zero variance; periodogram is 0; behavior undefined but no crash (may return grid endpoint) |
| White noise | `d` near 0, `H` near 0.5 |

## Cross-check tests

- `test_noise_d_near_zero` — 4096-point Gaussian noise: `|d| < 0.15`
- `test_hurst_offset` — `H = d + 0.5` identity within 1e-9
- `test_too_short_rejected` — 100-point input raises `KuantValueError`
- `test_se_positive` — reported SE strictly positive

## References

- Robinson, P. M. (1995). "Gaussian semiparametric estimation of long
  range dependence." Annals of Statistics 23, 1630-1661.
- Kunsch, H. R. (1987). "Statistical aspects of self-similar
  processes." Proc. First World Congress of the Bernoulli Society 1,
  67-74. (Original local-Whittle idea.)

## Related

- `kuant.stats.higuchihurst`, `kuant.stats.wavelethurst`,
  `kuant.stats.hurstrs` — nonparametric Hurst siblings; cross-check
  cheaply
- `kuant.stats.mfdfa` — multifractal generalization; use when
  monofractal `d` is not enough
- `kuant.stats.spectralentropy` — related frequency-domain summary

# higuchihurst — Higuchi fractal dimension Hurst estimator

## Purpose

Higuchi (1988) fractal-dimension method for the Hurst exponent.
Estimates `D` via the `k`-step curve length, then reports
`H = 2 - D`.

Position in the Hurst family:

- More stable than R/S on short series (200-500 points).
- Less sensitive to non-stationarity than R/S.
- Faster: `O(N * k_max)` vs R/S's `O(N * n_windows)`.

Best used as one of several cross-checked estimators. Disagreement
across `hurstrs`, `higuchihurst`, and `wavelethurst` is itself a
signal about the series' regularity.

## Public API

```python
from kuant.stats import higuchihurst

r = higuchihurst(x, k_max=30)
```

- `x` — 1D array. Non-finite values dropped.
- `k_max` — max step size in the curve-length calculation. Default 30,
  auto-capped to `n // 4` when the caller uses the default and `n < 120`.
- Returns `HiguchiHurstResult(hurst, fractal_dim, log_k, log_L, intercept, k_max)`.

## Design decisions

### 1. Curve length across `k` step sizes

For each step `k` and offset `m in [1, k]`, form the sub-series
`x[m-1], x[m-1+k], x[m-1+2k], ...` and compute

```
L_m(k) = sum |diff(sub)| * (n - 1) / ((n_pts - 1) * k^2)
```

The `k`-step curve length is `L(k) = mean_m L_m(k)`. Higuchi's
theorem: `L(k) ~ k^{-D}` for a fractal series, so
`log L(k) = -D * log k + const` and the OLS slope gives `D`.

`H = 2 - D` places `H in [0, 1]`; note that H = 0 corresponds to
white noise under Higuchi's convention, not H = 0.5 as under R/S.

### 2. Default `k_max = 30` with auto-cap for short inputs

Explicit `k_max` goes through `require_range(k_max, lo=4, hi=n//4)`
and raises `KE-VAL-RANGE` if out of bounds:

```
KE-VAL-RANGE: raised via require_range when k_max explicit
```

Only when the caller leaves the default of 30 does kuant clamp
`k_max = min(30, n // 4)`. This keeps the default feasible for
`n = 100..119`. Users who really want `k_max = 100` on `n = 200`
still get a clear error.

### 3. Empty step-count guard

For `k` near `n`, offsets `m > n - k` produce sub-series with < 2
points; those are skipped. If ALL offsets produce empty sub-series
the log value is set to NaN.

### 4. Log-log fit only on positive `L(k)`

`valid = isfinite(L_k) & (L_k > 0)`. Below 4 valid points the OLS
slope is unreliable:

```
KE-VAL-MIN-CLEAN: "insufficient valid log-log points; increase 'k_max'
or provide more data."
```

### 5. `n >= 100` floor

Higuchi is stable on short series, but below 100 the slope estimate is
too noisy to be useful:

```
KE-VAL-MIN-CLEAN: "only {n} finite values; need at least 100."
```

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| Fewer than 100 finite values | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| Explicit `k_max` outside `[4, n//4]` | raises `KuantValueError` `KE-VAL-RANGE` |
| Default `k_max = 30` with `n < 120` | auto-clamped to `n // 4` |
| Fewer than 4 valid log-log points | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| Monotone ramp | `D` near 1, `H` near 1 |
| White noise | `D` near 2, `H` near 0 |

## Cross-check tests

- `test_noise_dim_near_two` — 2000-point Gaussian noise: `D > 1.7`,
  `|H| < 0.3`
- `test_monotone_low_dim` — linear ramp: `D < 1.3`
- `test_too_short_rejected` — 50-point input raises `KuantValueError`
- `test_bad_kmax_rejected` — `k_max = 100` on `n = 200` raises

## References

- Higuchi, T. (1988). "Approach to an irregular time series on the
  basis of the fractal theory." Physica D 31, 277-283.

## Related

- `kuant.stats.hurstrs` — classical R/S estimator; different
  self-similarity convention (`H = 0.5` for noise)
- `kuant.stats.wavelethurst` — Abry-Veitch estimator; use as
  second-opinion cross-check
- `kuant.stats.localwhittle` — parametric long-memory alternative
- `kuant.stats.dfa` — detrended fluctuation analysis; monofractal
  cousin of `mfdfa`

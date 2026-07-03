# dfa — Detrended Fluctuation Analysis

## Purpose

Estimate the self-similarity exponent H via Peng et al.'s (1994) DFA
method. More robust than R/S Hurst (`hurstrs`) under nonstationary mean.

## Algorithm

1. Cumulative-sum the mean-centered series: `Y(k) = Σ (x_i - mean)`.
2. For each window size `w`:
   - Split `Y` into non-overlapping windows of length `w`
   - Fit a linear trend within each window; keep the residual
   - `F(w)` = root mean square of residuals across windows
3. Regress `log F(w)` on `log w`. Slope = H.

Interpretation:
- `H ≈ 0.5` — random walk / uncorrelated increments
- `H > 0.5` — persistent / trending
- `H < 0.5` — antipersistent / mean-reverting
- `H > 1` — nonstationary series (already integrated)

## Public API

```python
from kuant.stats import dfa

result = dfa(x, min_w=10, max_w=None, n_windows=20)
print(result.H, result.summary())
```

## When to prefer DFA over R/S Hurst

- Underlying trend or nonstationary drift you can't remove
- Longer windows where R/S estimator's small-sample bias hurts
- Log-log regression stability over multiple scales

## Design decisions

### Vectorized linear detrend per window

Within each anchor's window, we vectorize the OLS detrend across the
`n_windows_here` non-overlapping tiles: cov(t, Y) / var(t) gives the
slope for every tile in one call.

### Log-spaced window sizes

`w_min` up to `n // 4` on a log scale. Small end catches short-range
correlations; large end catches long-memory.

## Related

- `hurstrs` — cousin R/S estimator (simpler, less robust)
- `rollhurst` — rolling R/S variant

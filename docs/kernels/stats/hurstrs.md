# hurstrs — Hurst exponent via rescaled-range (R/S) analysis

## Purpose

Estimate the Hurst exponent `H` of a stationary series by regressing
the log of the mean rescaled range against the log of the window
size:

```math
\text{log } \overline{R/S}(w) \approx H \cdot \text{log } w + c
```

`H = 0.5` on a martingale; `H > 0.5` on a persistent / trending
series; `H < 0.5` on an antipersistent / mean-reverting series.

Originally from Hurst (1951, reservoir sizing on the Nile); brought
into finance by Mandelbrot & Wallis (late 1960s) and revisited
periodically since.

## Public API

```python
from kuant.stats import hurstrs

result = hurstrs(r, min_w=10, max_w=None, n_windows=20)
print(result.summary())
```

- `r` — 1D array of returns (or any stationary series). NaN-tolerant.
- `min_w` — smallest window size in the log-log regression.
- `max_w` — largest window size. Defaults to `len(r) // 4`.
- `n_windows` — approximate number of log-spaced windows; the true
  count after integer dedup is in `result.n_windows`.

Returns `HurstResult` with fields:

- `H` — scalar Hurst estimate
- `windows` — 1D int, window sizes used
- `log_rs` — 1D float, mean `log(R/S)` at each window (NaN where the
  window produced no valid samples)
- `intercept` — regression intercept
- `n_windows` — number of distinct windows

## Design decisions

### Composes existing primitives

Range of the detrended cumulative series comes from
`kuant.stats.rollrange`; the standard deviation comes from
`kuant.stats.rollstd`. No new inner-loop code paths.

### Log-spaced windows

Windows are chosen on a log scale between `min_w` and `max_w` so the
regression fits a linear log-log relationship with roughly equal
weight per decade. Integer dedup means the effective window count is
`<= n_windows`; the true count is reported.

### NaN policy

NaNs in the input propagate through `rollrange` and `rollstd` and
are dropped at the per-window averaging step. A window whose R/S
samples are all non-finite or non-positive contributes NaN to the
regression and is excluded from the log-log fit. If fewer than three
windows survive, a `ValueError` is raised so the caller can widen
the window range or the input length.

### CPU-only compute

R/S is a sequence of small window-scans and a single OLS fit. CuPy
inputs are converted to numpy at the boundary. The rolling wrapper
`rollhurst` is where the compute cost lives, and it is still
CPU-bound for typical trailing-window sizes.

## Related tools

- `kuant.stats.rollhurst` — trailing-window rolling version
- `kuant.stats.rollrange`, `kuant.stats.rollstd` — the primitives
  this composes

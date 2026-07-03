# rollhurst — Rolling Hurst exponent

## Purpose

Fit a Hurst exponent via `hurstrs` on each trailing window of length
`window`. Produces a time series `H_t` of the same length as the
input, with the first `window - 1` entries NaN.

Useful when the underlying series may exhibit regime-varying
self-similarity — e.g. periods of persistence interleaved with
periods of mean-reversion.

## Public API

```python
from kuant.stats import rollhurst

H_t = rollhurst(r, window=252, min_w=8, n_windows=8)
```

- `r` — 1D array of returns.
- `window` — trailing-window length in bars.
- `min_w`, `n_windows` — passed through to `hurstrs` at each anchor.
  Kept small by default so the inner R/S fit can succeed on the
  shorter trailing segments used here.

Returns a 1D `numpy.ndarray` of the same length as `r`. `H_t[t]` is
the R/S Hurst estimate over `r[t - window + 1 : t + 1]`; NaN for
`t < window - 1` or when the inner fit could not complete.

## Design decisions

### Trailing loop, not a compiled inner kernel

R/S is not trivially vectorizable across anchors because each anchor
runs its own log-log OLS. The naive `O(n · window)` loop is what
this implements. For heavy-use scans, cache the result rather than
recomputing.

### Small inner window defaults

`min_w=8` and `n_windows=8` are chosen so the inner R/S fit still
has enough distinct window sizes even when `window` is only a few
hundred bars. Callers with much longer windows can raise
`n_windows` to reduce log-log noise.

### NaN pass-through

Inner failures (short segment, degenerate range, or fewer than three
finite log-log points) return NaN at that anchor and do not raise —
the wrapper is expected to run over many anchors and callers filter
non-finite values downstream.

## Related tools

- `kuant.stats.hurstrs` — the underlying single-window estimator

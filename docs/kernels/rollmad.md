# rollmad — Rolling median absolute deviation

## Purpose

`rollmad(x, w)[i]` = median absolute deviation of `x[i-w+1:i+1]` from
its own median:

```math
mad = median(|x_j - median(x_window)|)
```

Robust dispersion measure — insensitive to outliers, unlike standard
deviation. Foundation for robust z-scores, outlier gating that
doesn't over-react to a single spike.

## Public API

```python
from kuant.stats import rollmad
result = rollmad(x, window)
```

## Design decisions

### Sliding-view + double xp.median

```python
windowed  = sliding_window_view(x, w)              # (n-w+1, w)
center    = xp.median(windowed, axis=1, keepdims=True)
deviations = xp.abs(windowed - center)
mad        = xp.median(deviations, axis=1)
```

Two axis-1 medians per window. O(n·w log w) per level. No cumsum
possible because medians don't decompose additively.

### Strict NaN policy for free

`xp.median` propagates NaN by default, so windows with any NaN in the
input give NaN output. No explicit mask needed.

### No scale-factor conversion

Some libraries return `1.4826 · MAD` (an estimate of σ under a normal
distribution). We return raw MAD; users can multiply as needed.

## Cross-check tests

- Golden: median(|[1,2,3,100,5] - 3|) = median([2,1,0,97,2]) = 2
- Robustness: with one 1000.0 outlier in a uniform stream, MAD stays
  small (~1) while rollstd blows up (~100)
- Constant window → 0

## Test coverage (3 tests)

Golden, outlier robustness, constant handling.

## Direct usage in kuant

- Robust z-score: `(x - median) / mad` (with sign-preserving properties)
- Outlier gates that ignore transient single-bar spikes
- Volatility estimator resistant to fat-tail contamination

## Related kernels

- `kuant.stats.rollquantile` — same sliding-view family
- `kuant.stats.rollstd` — non-robust dispersion counterpart

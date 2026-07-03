# rolltailindex — Rolling Hill tail index

## Purpose

Apply `tailindex` to each trailing window of length `window`. Result
is a time series ξ_t; rising values signal a fattening left tail.

## Public API

```python
from kuant.stats import rolltailindex

xi_t = rolltailindex(x, window, k_frac=0.10, min_k=10)
```

## Uses

- Regime detection: rising ξ_t often precedes stress periods
- Tail-hedge sizing: scale hedge with observed ξ_t
- Cross-strategy tail comparison: compare ξ_t across sleeves

## Related

- `tailindex` — single-window Hill estimator

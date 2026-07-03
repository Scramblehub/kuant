# tailindex — Hill tail-index estimator

## Purpose

Estimate the tail index ξ of the top-k order statistics via Hill's
(1975) log-ratio formula:

```math
\hat\xi = \frac{1}{k} \sum_{i=1}^k \bigl( \log X_{(i)} - \log X_{(k+1)} \bigr)
```

`ξ > 0` → Pareto-like heavy tail; `ξ = 0` → exponential; `ξ < 0` → bounded.

## Public API

```python
from kuant.stats import tailindex

xi = tailindex(x, k_frac=0.10, min_k=10)
```

- `x` — positive values (typically loss magnitudes).
- `k_frac` — fraction of the sample used as the tail.
- `min_k` — absolute floor on the tail size.

## Design decisions

### Sensible defaults for common quant use

`k_frac=0.10` and `min_k=10` fit typical fat-tail loss series (~10-15%
tail). For very long series, drop k_frac to 0.05 to focus on the
extreme tail.

### Non-positive / NaN filtered

Only finite positive values enter the ranking. If fewer than `min_k+2`
survive, returns NaN.

### Known bias on non-Pareto tails

Hill is only consistent on regularly-varying tails. For pure exponential
you get a biased estimate ~0.25-0.35 instead of 0. Compare across
samples to rank tail-heaviness rather than take absolute values as
gospel.

## Related

- `rolltailindex` — rolling variant
- `hurstrs` — self-similarity exponent (different tail primitive)

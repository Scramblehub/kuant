# logtccdf — Numerically stable log of upper-tail Student-t

## Purpose

Compute `log(1 - tcdf(x, df)) = log(tcdf(-x, df))` without underflow
in the deep right tail.

Naive `log(1 - tcdf(x, df))` underflows to `-inf` for large positive x.
Wraps `logtcdf(-x, df)` which handles the extreme case via asymptotic
fallback.

## Public API

```python
from kuant.core import logtccdf

result = logtccdf(x, df)
```

## Uses

- Fat-tail VaR: `log(P(loss > x))` in log space
- Tail-loss expectation integrals
- p-value computation for right-tailed Student-t tests
- Any workflow needing stable upper-tail probability

## Related

- `kuant.core.logtcdf` — the underlying primitive
- `kuant.core.lognormccdf` — Gaussian analog
- `scipy.stats.t.logsf` — reference implementation

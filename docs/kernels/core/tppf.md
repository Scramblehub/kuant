# tppf — Student-t inverse CDF

## Purpose

Given `p ∈ (0, 1)` and degrees of freedom `df`, return `x` such that
`tcdf(x, df) = p`.

## Public API

```python
from kuant.core import tppf

x = tppf(p, df)
```

Returns:
- Quantile x for `0 < p < 1`
- `-inf` for `p = 0`
- `+inf` for `p = 1`
- `nan` for `p ∉ [0, 1]`, `p = nan`, `df ≤ 0`, or `df = nan`

## Uses

- Fat-tail VaR: `VaR(α) = μ + σ·tppf(α, df)` where σ is scale.
- Confidence intervals for t-distributed test statistics.
- Fat-tail Monte Carlo sampling via inverse-transform.

## Design decisions

### scipy backend

`stdtrit` (scipy.special) is the reference implementation. No cupy
equivalent exists in `cupyx.scipy.special`, so cupy input takes the
H↔D fallback path in `_special_bridge`. GPU-heavy workloads may prefer
Newton polish on `tcdf` (higher throughput at the cost of accuracy).

### Sentinel values

Boundary and out-of-range handling matches `normppf`:
- `p = 0` → `-inf`
- `p = 1` → `+inf`
- `p` outside `[0, 1]` or invalid df → `nan`

## Related

- `tcdf`, `tpdf`
- `scipy.stats.t.ppf`
- `kuant.core.normppf` — Gaussian analog

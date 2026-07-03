# moneynessbucket — Classify options by moneyness

## Purpose

Assign each option in a chain an integer bucket label based on
log-forward moneyness:

```math
m = \ln\left(\frac{K}{F}\right)
\quad\text{where}\quad F = S \cdot e^{(r-q)T}
```

## Public API

```python
from kuant.options import moneynessbucket

buckets = moneynessbucket(S, K, T, r, q=0.0, edges=None)
```

Broadcasts S, K, T, r, q to common shape. `edges` is a 1D array of
break-points; default is `[-0.10, -0.03, 0.03, 0.10]` giving 5 buckets:

- 0: `m < -0.10` — deep ITM call side / deep OTM put side
- 1: `-0.10 ≤ m < -0.03` — ITM call / OTM put
- 2: `-0.03 ≤ m < 0.03` — ATM (near forward parity)
- 3: `0.03 ≤ m < 0.10` — OTM call / ITM put
- 4: `m ≥ 0.10` — deep OTM call / deep ITM put

## Design decisions

### Log-FORWARD moneyness, not log-spot moneyness

`m = ln(K/F)` uses the risk-neutral forward, not spot. This makes the
buckets more natural for options math:
- BS d1, d2 depend on `ln(S/K) + (r-q)T` = `-m`
- `m = 0` means K = F, the natural ATM
- Rate/dividend adjustments are absorbed into F, not the bucket edges

If you want log-spot moneyness `ln(K/S)`, pass `r=0` and `q=0`.

### `np.digitize` semantics

Bins are `[bins[i-1], bins[i])`. Elements equal to a bin edge go to the
right bin. See `test_exactly_on_edge`.

### Custom edges pass-through

Pass any monotone 1D `edges` array to customize. Common choices:
- `[-0.05, 0.05]` for tight-ATM screening
- `[-0.20, -0.10, 0.10, 0.20]` for coarse ITM/ATM/OTM split
- Per-vol-regime edges scaled by `σ√T` for standardized moneyness

## Related

- `deltabucket` — analogous but on delta axis
- `bscall`, `bsput` — price at each moneyness

# rollsum — Rolling window sum

## Purpose

`rollsum(x, w)[i]` = sum of `x[i-w+1 : i+1]`.

Distinct primitive from `rollmean` because volume totals, transaction
counts, and event aggregations want the raw sum, not the average.

## Public API

```python
from kuant.stats import rollsum
result = rollsum(x, window)
```

## Design decisions

Same cumsum trick as `rollmean` — one O(n) forward pass and differences.
Strict-window NaN policy (parallel cumsum on the NaN indicator).

The only meaningful difference from `rollmean` is the missing `/w` at
the end. Kept as a distinct kernel for API clarity and to save the
elementwise multiply if a caller wanted to convert back.

## Edge cases

| Condition | Output |
|---|---|
| `window == 1` | identity |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| NaN in window | NaN |

## Test coverage (10 tests)

Golden, pandas reference (uniform + with NaNs), consistency
(`rollsum == rollmean * w`), edge cases, dtype preservation,
CPU==GPU parity.

## Related kernels

- `kuant.stats.rollmean` — sibling, `rollsum / w`
- `kuant.stats.rollstd`, `kuant.stats.rollcorr` — same cumsum pattern

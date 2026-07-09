# drawdown: Peak-to-trough series and max drawdown

## Purpose

For a running equity curve, compute the drawdown at every bar:

```math
drawdown[t] = equity[t] / max(equity[0..t]) - 1
```

Always `<= 0`. Zero at every new peak; negative in between. `max_dd`
is the most-negative value across the whole series, together with
its peak and trough positions and whether the curve recovered.

For a trailing-window version use `kuant.stats.rollmdd`. This kernel
is the full-history one.

## Public API

```python
from kuant.portfolio import drawdown
import numpy as np

equity = np.cumprod(1 + returns)
r = drawdown(equity)
r.series           # (T,) drawdown per bar, in [-1, 0]
r.max_dd           # scalar, negative
r.peak_position    # index of peak preceding max_dd
r.trough_position  # index of trough
r.duration         # trough - peak in bars
r.recovered        # bool: curve reached peak again after trough
print(r.summary())
r.to_parquet("drawdown.parquet")
```

- `equity` — 1D. Must be strictly positive. Use
  `np.cumprod(1 + returns)` on simple returns or
  `np.exp(np.cumsum(log_returns))` on log returns.

## Design decisions

### 1. NaN-safe running max via fill-with-`-inf`

`np.maximum.accumulate(np.where(finite, equity, -inf))` lets NaN bars
propagate without corrupting the running peak. The `-inf` fill means
NaN bars neither update the max nor register a drawdown; the ratio
step then yields NaN at those positions.

### 2. Peak position is the FIRST index at the peak value

After finding the trough with `nanargmin`, the peak value is
`running_max[trough]`. `np.argmax(equity == peak_val)` returns the
first index equal to that peak, which is the correct "peak that fed
this trough" when the curve visited the same peak multiple times.

### 3. `recovered` semantics

`True` iff any equity value AFTER the trough reaches `>= peak_val`.
`False` when the curve is still under water at the end of the
series, matching the "underwater at cutoff" state that portfolio
managers care about.

### 4. Strict positivity check

Zero or negative equity breaks the peak/max ratio semantics
(division-by-zero or sign-flipped drawdown). Raises
`KuantValueError [KE-VAL-POSITIVE]` with a fix hint pointing at
`np.cumprod(1 + returns)`.

### 5. All-NaN input warns rather than errors

Returns a NaN series and `max_dd = NaN` after emitting
`KW-DRAWDOWN-ALL-NAN`. Consuming pipelines typically want a
propagated NaN rather than an exception on the "no data" case.

### 6. Parquet writes only the series

`to_parquet` serializes the `(T,)` drawdown column. Scalar summary
fields are cheap to recompute and not written.

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| Empty `equity` | `KuantValueError [KE-VAL-EMPTY]` |
| Any finite `equity[i] <= 0` | `KuantValueError [KE-VAL-POSITIVE]` |
| Non-1D input | raised by `require_1d` |
| All-NaN input | `KuantNumericWarning [KW-DRAWDOWN-ALL-NAN]`, `max_dd = NaN` |
| Curve never draws down | `max_dd = 0`, `peak = trough = 0`, `recovered = False` |
| Trough at final bar | `recovered = False` |
| `pyarrow` missing at `to_parquet` | raises via `require_dep` |

## Cross-check tests

- Golden peak/trough on hand-computed 7-bar curve
  (`[100, 105, 110, 100, 88, 95, 105]` gives `max_dd = -0.20`,
  peak_pos = 2, trough_pos = 4, `recovered = False`).
- NaN propagation through running max.
- Parquet round-trip.

`tests/portfolio/test_drawdown.py`.

## References

- Standard definition. No specific literature citation.

## Related kernels

- `kuant.stats.rollmdd` — rolling max drawdown over a trailing
  window.
- `kuant.portfolio.riskmetrics.ulcer_index` — RMS drawdown; captures
  duration alongside depth.
- `kuant.portfolio.riskmetrics.drawdown_table` — episode-level view
  with multiple ranked drawdowns.

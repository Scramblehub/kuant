# rollema — Exponentially weighted moving average

## Purpose

```math
ema[0] = x[0]
ema[i] = α · x[i] + (1 - α) · ema[i-1]
```

Recursively-smoothed baseline. Widely used for signal smoothing,
alpha decay in trading heuristics, adaptive baselines that respond
faster to recent data than a fixed-window mean.

## Public API

```python
from kuant.stats import rollema

result = rollema(x, span=None, alpha=None)
```

Exactly ONE of `span` or `alpha` must be provided:

- `span` — pandas convention: `alpha = 2 / (span + 1)`. Must be `>= 1`.
- `alpha` — smoothing factor directly. Must be in `(0, 1]`.

## Design decisions

### 1. Different algorithmic pattern from the rest of `kuant.stats`

Not window-based. The recurrence has an inherent sequential
dependency — you can't compute `ema[i]` without `ema[i-1]`. No
cumsum trick, no sliding-view.

### 2. CPU path uses `scipy.signal.lfilter`

Implemented as an IIR filter:

```math
b = [α], a = [1, -(1-α)]
```

`lfilter` runs the recurrence in compiled C, avoiding the Python-loop
overhead that would make plain `for i in range(n)` unacceptably slow
for large inputs.

Initial condition `zi = (1 - α) · x[0]` chosen so `ema[0] = x[0]`.

### 3. GPU path — CPU fallback for V1

For cupy input, we transfer to numpy, run the CPU path, and transfer
back. A proper GPU implementation would need a parallel prefix scan
adapted to the linear recurrence (using the Blelloch scan pattern or
similar). Not implemented in V1.

Cost: one `.get()` and one `xp.asarray()` per call. For
research-workload sizes (< 1M elements), acceptable.

### 4. `span` matches pandas convention

pandas' `ewm(span=w)` uses `alpha = 2/(w+1)`. Same formula here so
`rollema(x, span=w)` matches `pd.Series(x).ewm(span=w, adjust=False).mean()`
bit-for-bit.

### 5. `adjust=False` semantics

We use the recursive form, matching pandas' `adjust=False`. The
`adjust=True` variant uses a different weighted-average formula
that's more accurate at the start of the series but requires
different math. Not implemented in V1.

## Edge cases

| Condition | Output |
| --- | --- |
| Empty input | empty output |
| `span < 1` | raises `ValueError` |
| `alpha` outside `(0, 1]` | raises `ValueError` |
| Neither / both of span/alpha given | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| Constant input | constant output (equal to input) |
| `alpha == 1` (span == 1) | identity (no smoothing) |
| NaN in input | NaN propagates through the recurrence |

## Cross-check tests

- `test_recursion_manual_verification` — matches a Python-loop
  reimplementation of the recurrence
- `test_matches_pandas_ewm` — 500 random samples across 4 alphas
- `test_span_matches_pandas` — via the `span → alpha` conversion

## Test coverage (17 tests)

Golden (recursion property, constant input stays constant),
pandas reference (alpha + span variants), edge cases (input
validation, empty, dtype), qualitative (low alpha = high smoothing),
CPU==GPU parity.

## Direct usage in kuant

- Signal smoothing before further processing (e.g. smoothed returns
  as input to zscore)
- Alpha decay in position sizing (recent returns weighted more)
- Adaptive volatility baseline (ema of squared returns)
- Fast/slow crossovers for regime detection

## Related kernels

- `kuant.stats.rollmean` — arithmetic-mean baseline (uniform weights)
- **Future**: `kuant.stats.rollemvol` — exponentially weighted variance
  and volatility, matches pandas `.ewm().var()` / `.std()`
- **Future**: Native GPU implementation via parallel scan

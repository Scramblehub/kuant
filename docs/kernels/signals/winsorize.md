# winsorize — Quantile-capped values

## Purpose

Clip values in an array to a lower and upper quantile of that same
array. The one primitive every signals desk re-implements.

For 2D input, two modes:

- **`per_row=True`** (default): cross-sectional. Each row (each time
  slice) uses its own quantiles. Standard for factor scores that span
  names at a given date.
- **`per_row=False`**: time-series. Each column (each name) uses its
  own quantiles from its full history. For per-name noise clipping.

NaN cells are excluded from the quantile computation and preserved as
NaN in the output.

## Public API

```python
from kuant.signals import winsorize

y = winsorize(x, lo=0.01, hi=0.99, per_row=True)
```

- `x`: 1D or 2D array. Integers promote to float64 so NaN can be
  represented.
- `lo`, `hi`: probabilities in `[0, 1]` with `lo < hi`. Values below
  the `lo`-th quantile clip up; values above the `hi`-th quantile
  clip down.
- `per_row`: 2D-only. `True` cross-sectional (rows), `False`
  time-series (columns).

Example:

```python
>>> import numpy as np
>>> from kuant.signals import winsorize
>>> x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100])
>>> winsorize(x, lo=0.0, hi=0.9).tolist()
[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.1]
```

## Design decisions

### 1. NaN excluded from quantiles, preserved in output

`np.nanquantile` computes the cut points ignoring NaN. The final
`np.clip` only touches finite entries, so NaN passes through as NaN.
Winsorization is a value-scale operation; missingness is orthogonal
and should not be manufactured by a cap.

### 2. Strict `lo < hi`, both in `[0, 1]`

`lo >= hi` (including equal) raises `KuantValueError` with
`[KE-VAL-RANGE]`. Range validation on each bound flows through
`require_probability`. Equal bounds would collapse the distribution
to a constant, which is almost never the intent; if it is, the caller
should say so directly (fillna with the median, etc.).

### 3. Aggressive-limits warning at `lo > 0.25` or `hi < 0.75`

Above `lo = 0.25` (or symmetrically `hi < 0.75`) the boundary values
touch or cross the median, so more than half the distribution is
being pinned to a boundary. Emits `KuantNumericWarning` with
`[KW-WINSORIZE-AGGRESSIVE-LIMITS]`. Typical usage is `(0.01, 0.99)`
or `(0.05, 0.95)`; the warning nudges users away from silently
destroying the signal.

### 4. Per-row / per-column via 1D helper loop

`_winsorize_1d` handles the actual work. 2D dispatch loops over rows
or columns. An axis-aware `np.nanquantile` broadcast would vectorize
this, but the loop is O(T + N) function-call overhead against a large
inner cost, and the loop is easier to reason about around all-NaN
rows.

### 5. All-NaN row / column passes through unchanged

If a row (or column, per mode) has no finite entries, `np.nanquantile`
would warn and return NaN. The `_winsorize_1d` fast-path returns the
input unchanged in that case, avoiding a spurious warning and keeping
the shape stable.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim not in (1, 2)` | `KuantShapeError` `[KE-SHAPE-EXPECTED]` |
| `lo` or `hi` outside `[0, 1]` | `KuantValueError` via `require_probability` |
| `lo >= hi` | `KuantValueError` `[KE-VAL-RANGE]` |
| `lo > 0.25` or `hi < 0.75` | `KuantNumericWarning` `[KW-WINSORIZE-AGGRESSIVE-LIMITS]` |
| All-NaN row / column | Passes through unchanged |
| Integer input | Promotes to float64 |

## Related kernels

- `kuant.signals.neutralize`: residualize a signal against factor
  exposures. Winsorize the raw signal first, then neutralize.
- `kuant.signals.factor_ic`: consumes the cleaned cross-sectional
  panel this kernel typically produces.
- `kuant.stats.zscore`: pair upstream (or downstream) for scale
  standardization.

## References

- Dixon, W. J. (1960). Simplified estimation from censored normal
  samples. *Annals of Mathematical Statistics*, 31(2), 385-391.

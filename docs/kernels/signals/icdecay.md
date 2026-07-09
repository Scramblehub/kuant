# icdecay — Spearman IC decay across forecast horizons

## Purpose

For each horizon `h`, compute the Spearman rank correlation between
the signal at time `t` and the cumulative forward return over
`[t + 1, t + h]`. That correlation is the Information Coefficient
(IC) at horizon `h`. Sweeping `h` traces the DECAY CURVE: how quickly
the signal's edge evaporates as the forecast reaches further out.

Two per-horizon outputs:

- **`ic`**: the Spearman correlation itself. Positive means "high
  signal now implies high forward return". Rule of thumb: `|IC| > 0.02`
  is real, `|IC| > 0.05` is very good.
- **`ic_tstat`**: `ic` divided by its approximate standard error
  `1 / sqrt(n)`. `|t| > 2` is the rough threshold for
  "distinguishable from zero" at the tested sample size.

Univariate time-series. For the cross-sectional analogue (compare `N`
names at each date), see `kuant.signals.factor_ic`.

## Public API

```python
from kuant.signals import icdecay

result = icdecay(signal, forward_returns, horizons=(1, 5, 21, 63))
result.horizons                # tested horizons
result.ic                      # per-horizon Spearman IC
result.ic_stderr               # per-horizon 1/sqrt(n)
result.ic_tstat                # ic / ic_stderr
result.n                       # clean overlapping pairs per horizon
```

- `signal`: 1D array, length `T`. Signal value at each time.
- `forward_returns`: 1D array, length `T`. Periodic return between
  time `t` and `t + 1`. Log-returns preferred; simple-returns
  approximation degrades as `h` grows.
- `horizons`: sequence of positive ints, all strictly less than `T`.

Requires `scipy.stats.spearmanr`. Lazy import.

Example:

```python
>>> import numpy as np
>>> from kuant.signals import icdecay
>>> rng = np.random.default_rng(0)
>>> T = 500
>>> signal = rng.standard_normal(T)
>>> forward_ret = 0.02 * signal + rng.standard_normal(T) * 0.02
>>> r = icdecay(signal, forward_ret, horizons=(1, 5, 21))
>>> r.ic[0] > 0.2
True
```

## Design decisions

### 1. Cumulative-sum trick for horizon returns

Forward return over `[t + 1, t + h]` is a length-`h` sum. Naive
implementation costs `O(T * h)` per horizon. We prepend a zero to
`cumsum(forward_returns)` and get every horizon window in one vector
op: `fwd[t] = csum[t + h + 1] - csum[t + 1]`. Same O(T) whether the
horizon is 1 or 252.

### 2. NaN handling via parallel NaN-count cumsum

We can't sum through NaNs (any NaN in the window pollutes the return).
Rather than guard each window individually, we run a second cumsum on
`isnan(forward_returns).astype(int64)`. The rolling NaN count tells
us which windows are clean. The signal NaN check adds one more mask.
Same trick used in `kuant.stats.rollmean`.

### 3. `1 / sqrt(n)` stderr, not a Fisher-z variance

Fisher-z gives a tighter interval on `rho` but the transform breaks
down near `|rho| = 1` and adds cognitive load for the caller.
`1 / sqrt(n)` is the classic conservative approximation, matches
what practitioners eyeball, and matches the `factor_ic` t-stat
convention across kuant.

### 4. Noise-floor warning at `abs(ic) < ic_stderr`

If the IC at any tested horizon is smaller in magnitude than its own
standard error, we emit `KuantNumericWarning` with
`[KW-IC-NOISE-FLOOR]` citing the first offending horizon. This is
the "indistinguishable from noise at this sample size" failure mode,
the one most likely to make a sales pitch disappear on real capital.

The warning does not raise; the result is still returned for the
caller to inspect.

### 5. No-clean-pairs safety warning

If EVERY horizon fails to produce 3+ clean overlapping pairs (`n < 3`
means Spearman is undefined), we emit `KuantNumericWarning` with
`[KW-ICDECAY-NO-CLEAN]`. Typical cause: a NaN block covering the
signal window, or horizons that consume the entire history.

### 6. Horizon bound: `h < T`

`h >= T` leaves no overlapping observations. Raises `KuantValueError`
with `[KE-VAL-RANGE]`. The bound is strict; even `h == T - 1` gives
one pair, which is below the Spearman minimum but is left to the
noise-floor and `n < 3` guards downstream.

## Return shape

**ICDecayResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `horizons` | 1D int64 array | Horizons tested |
| `ic` | 1D float array | Spearman IC per horizon, NaN if `n < 3` |
| `ic_stderr` | 1D float array | `1 / sqrt(n)` per horizon |
| `ic_tstat` | 1D float array | `ic / ic_stderr` per horizon |
| `n` | 1D int64 array | Clean overlapping pairs per horizon |

`.summary()` prints an aligned table plus the peak `|IC|` horizon.
`.to_parquet(path)` writes the curve to parquet (requires pyarrow;
missing dep raises `KuantValueError` with `[KE-DEP-MISSING]`).

## Edge cases

| Condition | Behavior |
| --- | --- |
| `signal` or `forward_returns` not 1D | `KuantValueError` via `require_1d` |
| Length mismatch | `KuantValueError` via `require_equal_length` |
| `horizons` empty | `KuantValueError` `[KE-VAL-RANGE]` |
| Any `h <= 0` | `KuantValueError` via `require_positive` |
| Any `h >= T` | `KuantValueError` `[KE-VAL-RANGE]` |
| Horizon with `n < 3` clean pairs | `ic`, `ic_stderr`, `ic_tstat` NaN; `n` reported |
| All horizons NaN | `KuantNumericWarning` `[KW-ICDECAY-NO-CLEAN]` |
| Any horizon with `abs(ic) < ic_stderr` | `KuantNumericWarning` `[KW-IC-NOISE-FLOOR]` |
| scipy missing | `KuantValueError` `[KE-DEP-MISSING]` |

## Related kernels

- `kuant.signals.factor_ic`: cross-sectional per-date IC on a `(T, N)`
  panel (compare across names at each date, not across time).
- `kuant.signals.winsorize`, `kuant.signals.neutralize`: standard
  preprocessing on the signal before an IC study.
- `kuant.nulltest.bootstrap_ic`: block-bootstrap p-value and CI on
  an IC point estimate.

## References

- Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management:
  A Quantitative Approach for Producing Superior Returns and
  Controlling Risk*, 2nd ed. McGraw-Hill. (IC and IR framework.)
- Spearman, C. (1904). The proof and measurement of association
  between two things. *American Journal of Psychology*, 15(1), 72-101.

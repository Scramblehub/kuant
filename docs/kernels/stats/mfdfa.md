# mfdfa — Multifractal detrended fluctuation analysis

## Purpose

Multifractal DFA (Kantelhardt 2002). Extends classical DFA by
computing the generalized Hurst exponent `h(q)` across a range of
moment orders `q`. Diagnostic reads:

- `h(q)` constant in `q` -> monofractal (classical DFA / hurstrs
  suffice).
- `h(q)` varies with `q` -> multifractal. Width `max h(q) - min h(q)`
  quantifies the range of scaling behaviors present.

Financial return series are typically multifractal; the width is a
diagnostic of intermittency and long-range correlations.

## Public API

```python
from kuant.stats import mfdfa

r = mfdfa(x, q_values=None, scales=None, order=1)
```

- `x` — 1D array. Non-finite dropped. Requires `n >= 200`.
- `q_values` — moment orders. Default `[-3, -2, -1, 0.5, 1, 2, 3, 4]`.
  Note that `q = 2` recovers standard DFA.
- `scales` — segment sizes. Default log-spaced `[10, n // 4]`, 15
  unique integer points.
- `order` — polynomial detrending order (1 = linear). Must be in
  `[1, 5]`.
- Returns `MfdfaResult(q_values, h_q, multifractal_width, F_q_s,
  scales)`.

## Design decisions

### 1. Integrate first, detrend per segment

Following Kantelhardt (2002):

```
Y = cumsum(x - mean(x))
```

For each scale `s`, split `Y` into disjoint segments of length `s`,
polynomial-fit and subtract per segment (forward pass), then repeat
from the end (backward pass). The union of forward + backward
residual variances is the segment sample used for `F_q(s)`.

Backward pass matters when `n` is not a multiple of `s`; without it
the tail is discarded.

### 2. Moment aggregation and the `q = 0` limit

For `q != 0`:

```
F_q(s) = mean( var_seg ^ (q / 2) ) ^ (1 / q)
```

For `q -> 0` this diverges; L'Hopital gives the geometric-mean form,
implemented as `exp(0.5 * mean(log(var_seg)))`. The `|q| < 1e-8`
branch triggers this path so users can pass exactly 0.

### 3. Scaling law and `h(q)` extraction

Fit `log F_q(s) = h(q) * log s + const` by `polyfit` per `q`. Slope
is `h(q)`. Rows with < 4 valid `(scale, F_q)` points return
`h(q) = NaN`.

`multifractal_width = max(h_q) - min(h_q)` on the finite subset.
Constant `h(q)` -> width ~ 0.

### 4. Scale range guards `s < order + 2` and `s > n // 2`

A polynomial fit of order `p` on `p + 1` points is trivial; kuant
requires at least `p + 2` points per segment for a meaningful
residual variance. Segments longer than `n // 2` give only one
segment per pass, so `F_q(s)` collapses to a single value.

Any scale outside this range is silently skipped rather than
raising; the caller's default scales are always safe.

### 5. `order in [1, 5]` gate

`require_range(order, "order", lo=1, hi=5)` bounds polynomial degree.
Higher-order detrending removes stronger trends but wastes degrees of
freedom on short segments:

```
KE-VAL-RANGE: raised via require_range if order out of [1, 5]
```

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| `n < 200` | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| `order` outside `[1, 5]` | raises `KuantValueError` `KE-VAL-RANGE` |
| Scale `s < order + 2` | silently skipped |
| Scale `s > n // 2` | silently skipped |
| Fewer than 3 finite segment variances at scale `s` | row left as NaN |
| Fewer than 4 valid scales for a given `q` | `h(q) = NaN` |
| `q = 0` | geometric-mean branch (`exp(0.5 * mean(log var))`) |
| Monofractal input (e.g. iid noise) | width small; `h(2)` near 0.5 |

## Cross-check tests

- `test_noise_h2_near_half` — 2000-point Gaussian noise:
  `h(2) in (0.35, 0.65)`
- `test_multifractal_width_nonneg` — `width >= 0` on Gaussian
- `test_bad_order_rejected` — `order = 0` raises `KuantValueError`
- `test_too_short_rejected` — 100-point input raises `KuantValueError`

## References

- Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin,
  S., Bunde, A., Stanley, H. E. (2002). "Multifractal detrended
  fluctuation analysis of nonstationary time series." Physica A 316,
  87-114.
- Peng, C. K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H.
  E., Goldberger, A. L. (1994). "Mosaic organization of DNA
  nucleotides." Phys. Rev. E 49, 1685-1689. (Original DFA.)

## Related

- `kuant.stats.dfa` — monofractal q = 2 special case
- `kuant.stats.hurstrs`, `kuant.stats.higuchihurst`,
  `kuant.stats.wavelethurst`, `kuant.stats.localwhittle` — Hurst
  siblings
- `kuant.stats.spectralentropy` — a simpler complexity summary

# sortinoratio: Annualized full-history Sortino

## Purpose

Sortino replaces Sharpe's full-sample std with the DOWNSIDE-only
RMS deviation from a target:

```math
downside_std = sqrt(mean(min(returns - target, 0)^2))
sortino = (mean - target) * sqrt(ann_factor) / downside_std
```

Rationale: upside volatility is not risk. A strategy with heavy
right-tail returns should be rewarded, not penalized.

`target` is usually 0 (the Minimum Acceptable Return convention) or
a per-period risk-free rate.

For a trailing-window version use `kuant.stats.rollsortino`. This
kernel is the full-history scalar.

## Public API

```python
from kuant.portfolio import sortinoratio

r = sortinoratio(returns, ann_factor=252, target=0.0)
r.sortino          # annualized
r.mean_excess      # mean(returns - target)
r.downside_std     # RMS of min(returns - target, 0), OVER FULL SAMPLE
r.n
r.n_below_target   # count of observations below target
r.ann_factor
r.target
print(r.summary())
```

- `returns` — 1D. NaN is dropped.
- `ann_factor` — same convention as `sharperatio`.
- `target` — per-period Minimum Acceptable Return.

## Design decisions

### 1. Downside RMS over the FULL sample, not the downside subsample

`downside_std = sqrt(mean(downside * downside))` averages over all
`n` observations, not only the `n_below` that were actually below
target. This is the standard Sortino 1994 convention: it treats
zero-clipped upside observations as "no downside excursion this
bar" contributing zero to the RMS. Dividing by `n` instead of
`n_below` makes the Sortino directly comparable to Sharpe on the
same series.

### 2. No-downside case: `+inf`, `-inf`, or 0

`n_below == 0` means the downside RMS is zero and the ratio is
undefined. By convention:

- `mean_excess > 0` returns `+inf`.
- `mean_excess < 0` returns `-inf`.
- `mean_excess == 0` returns `0.0`.

`KW-SORTINO-NO-DOWNSIDE` fires either way so the caller can spot the
degenerate case in a tearsheet.

### 3. Tiny-downside guard emits `KW-SORTINO-TINY-DOWNSIDE`

If observations exist below target but their RMS is `< 1e-15`,
return `sortino = 0` and warn via `warn_zero_denominator`. Prevents
a huge nonsense ratio from FP-noise-scale downside.

### 4. Small-sample warning at `n < 30`

`KW-SORTINO-SMALL-SAMPLE`. Sortino has fatter sampling noise than
Sharpe because the downside subsample can be much smaller than the
full sample; the warning fires on the full-sample count, and the
underlying `n_below_target` is exposed in the result for a tighter
diagnosis.

### 5. NaN drop, not NaN propagate

Same policy as `sharperatio`: strip non-finite observations before
computing anything.

### 6. Target subtraction first

`excess = finite - target` before mean and downside. Passing
`target = 0` (default, the MAR convention) reproduces the raw
Sortino. Passing a per-period rf reproduces the "excess Sortino".

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| No finite returns | `KuantValueError [KE-VAL-FINITE]` |
| `ann_factor <= 0` | `KuantValueError` from `require_positive` |
| `n < 30` | `KuantNumericWarning [KW-SORTINO-SMALL-SAMPLE]` |
| `n_below_target == 0` | `KuantNumericWarning [KW-SORTINO-NO-DOWNSIDE]`, `sortino = +inf` / `-inf` / 0 by sign of `mean_excess` |
| Downside exists but `< 1e-15` RMS | `KuantNumericWarning [KW-SORTINO-TINY-DOWNSIDE]`, `sortino = 0` |
| Non-1D input | raised by `require_1d` |

## Cross-check tests

- Positive-drift Gaussian daily returns produce positive Sortino.
- No-downside synthetic (all-positive returns) fires
  `KW-SORTINO-NO-DOWNSIDE` and returns `+inf`.
- Symmetric Gaussian: Sortino close to Sharpe (both use `n` in the
  denominator).
- Small-sample warning fires below n=30.

`tests/portfolio/test_sortinoratio.py`.

## References

- Sortino & Price 1994, "Performance measurement in a downside risk
  framework," Journal of Investing 3(3).

## Related kernels

- `kuant.portfolio.sharperatio` — full-sample std counterpart.
- `kuant.stats.rollsortino` — trailing-window rolling Sortino.
- `kuant.portfolio.riskmetrics.omega` — a fatter-tail-friendlier
  alternative when Sortino is still too Gaussian.

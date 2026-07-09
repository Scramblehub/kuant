# sharperatio: Annualized full-history Sharpe

## Purpose

Full-history annualized Sharpe on a periodic return series:

```math
sharpe = (mean(returns) - rf) * sqrt(ann_factor) / std(returns)
```

For a trailing-window version use `kuant.stats.rollsharpe`. This
kernel is the full-history scalar.

## Public API

```python
from kuant.portfolio import sharperatio

r = sharperatio(returns, ann_factor=252, rf=0.0)
r.sharpe          # annualized
r.mean            # per-period excess-of-rf mean
r.std             # per-period sample std (ddof=1)
r.n               # finite-observation count
r.ann_factor
r.rf
print(r.summary())
```

- `returns` — 1D periodic returns. NaN is dropped.
- `ann_factor` — 252 daily, 52 weekly, 12 monthly, 1 already annual.
- `rf` — per-period risk-free rate (NOT annual). If you have an
  annual rf, divide by `ann_factor` before passing.

## Design decisions

### 1. Full-history convention

`mean` and `std` are computed over the whole finite subset. This
matches the classical Sharpe 1966/1994 definition; for rolling
Sharpe use `kuant.stats.rollsharpe`.

### 2. Sample std with `ddof=1`

`np.std(excess, ddof=1)` unless `n == 1`. Sample (Bessel-corrected)
std is the standard reporting convention. `ddof=1` also makes the
Sharpe estimator match the Bailey-Lopez de Prado probabilistic
Sharpe formula used in `kuant.portfolio.riskmetrics`.

### 3. NaN drop, not NaN propagate

Non-finite returns are silently removed. Rationale: any live
return-series has small numbers of NaN from data-vendor artifacts;
propagating them to a NaN Sharpe destroys downstream analytics for
a reason unrelated to strategy performance.

### 4. Zero-std guard emits `KW-SHARPE-CONSTANT-RETURNS`

If `std < 1e-15` (constant returns, or FP noise around a constant),
return `sharpe = 0` and emit the warning through
`warn_zero_denominator`. Dividing by FP-noise std would give a
nonsense huge Sharpe; zero is the honest answer for a constant
series.

### 5. Small-sample warning at `n < 30`

`KW-SHARPE-SMALL-SAMPLE` fires below 30 finite observations. The
standard error on Sharpe scales like `1/sqrt(n)`, so the estimate is
dominated by sampling noise below that scale. The warning nudges
users toward `probabilistic_sharpe` for a proper significance
statement.

### 6. Excess returns first, then mean and std

`excess = finite - rf` before both `mean` and `std`. This is the
Sharpe 1994 revision where risk-free is subtracted per bar. Passing
`rf = 0` (the default) reproduces the raw Sharpe.

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| No finite returns | `KuantValueError [KE-VAL-FINITE]` |
| `ann_factor <= 0` | `KuantValueError` from `require_positive` |
| `n < 30` | `KuantNumericWarning [KW-SHARPE-SMALL-SAMPLE]` |
| Constant returns (`std < 1e-15`) | `KuantNumericWarning [KW-SHARPE-CONSTANT-RETURNS]`, `sharpe = 0` |
| Non-1D input | raised by `require_1d` |
| `n == 1` | `std = 0`, warning fires, `sharpe = 0` |

## Cross-check tests

- 10-year synthetic daily with known mean/std reproduces Sharpe to
  1e-6.
- Constant-return input emits `KW-SHARPE-CONSTANT-RETURNS` and
  returns 0.
- Small-sample warning triggers below n=30.
- NaN drop parity: `sharperatio(r_with_nan) == sharperatio(r_clean)`
  when the only difference is the NaN rows.

`tests/portfolio/test_sharperatio.py`.

## References

- Sharpe 1966, "Mutual fund performance," Journal of Business 39(1).
- Sharpe 1994, "The Sharpe ratio," Journal of Portfolio Management
  21(1).

## Related kernels

- `kuant.portfolio.sortinoratio` — replaces `std` with downside-only
  RMS.
- `kuant.stats.rollsharpe` — trailing-window rolling Sharpe.
- `kuant.portfolio.riskmetrics.probabilistic_sharpe`,
  `deflated_sharpe` — significance and selection-bias adjustment on
  top of the raw Sharpe reported here.

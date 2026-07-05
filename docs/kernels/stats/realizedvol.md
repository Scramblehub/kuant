# realizedvol: OHLC volatility estimators and True Range

## Purpose

Five estimators from the microstructure literature that turn an OHLC
bar series into a per-bar volatility number, plus the True Range
primitive underlying practitioner stop-loss and sizing rules.

All estimators return per-bar standard deviation. Multiply by
`sqrt(ann_factor)` for annualized figures (same convention as
`rollstd`).

- `atr`: Wilder's True Range averaged over a window. Not a formal
  vol estimator; used as a range-based sizing input.
- `parkinson`: high/low only; Parkinson (1980).
- `garmanklass`: OHLC; Garman & Klass (1980).
- `rogersatchell` (`rogerssatchell`): OHLC; Rogers & Satchell (1991).
- `yangzhang`: OHLC with overnight-gap term; Yang & Zhang (2000).

## Public API

```python
from kuant.stats import atr, parkinson, garmanklass, rogerssatchell, yangzhang

a  = atr(high, low, close, window=14)              # -> 1D array
p  = parkinson(high, low)                          # -> float
gk = garmanklass(open_, high, low, close)          # -> float
rs = rogerssatchell(open_, high, low, close)       # -> float
yz = yangzhang(open_, high, low, close, prev_close=None)  # -> float
```

All price inputs are 1D arrays of equal length. Prices must be
strictly positive (each estimator applies `log(H/L)` or similar and
would otherwise emit NaN through the log).

## Design decisions

### 1. `atr` uses the SMA form, not Wilder's EMA

Wilder's original ATR (1978) recursively smooths True Range with a
Wilder-style EMA. The industry-standard closed-form variant is a
simple moving average of the True Range series over `window` bars.
kuant ships the SMA form because it is deterministic, reproducible
across languages, and matches what most modern reference libraries
compute.

True Range at bar `t`:

```math
TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
```

For `t = 0`, `C_{t-1}` is unavailable; we degenerate to `H_0 - L_0`.
This avoids seeding the series with NaN and matches TA-Lib behavior.

### 2. Parkinson: H/L only, driftless assumption

```math
sigma_P = sqrt( sum(ln(H/L)^2) / (4 * ln(2) * n) )
```

Parkinson (1980) derives this under a driftless geometric Brownian
motion. Fine for currencies and instruments without a persistent
trend; biased LOW when the underlying drifts (the daily range fails
to reflect the price displacement).

### 3. Garman-Klass: OHLC, still driftless

```math
sigma_GK^2 = mean( 0.5 * ln(H/L)^2 - (2 * ln 2 - 1) * ln(C/O)^2 )
```

Garman & Klass (1980). Improves on Parkinson by incorporating the
open and close. Assumes zero drift; biased LOW under drift for the
same structural reason.

### 4. Rogers-Satchell: OHLC, robust to drift

```math
sigma_RS^2 = mean( ln(H/O) * ln(H/C) + ln(L/O) * ln(L/C) )
```

Rogers & Satchell (1991). The identity cancels the drift term
analytically. Recommended when the series has a meaningful daily
drift (e.g. trending equities).

### 5. Yang-Zhang: best-in-class for gappy equities

```math
sigma_YZ^2 = sigma_ON^2 + k * sigma_OC^2 + (1 - k) * sigma_RS^2
```

with the Rogers-Satchell contribution `sigma_RS^2`, the overnight
log-return variance `sigma_ON^2 = Var(ln(O_t / C_{t-1}))`, the
open-to-close log-return variance `sigma_OC^2 = Var(ln(C_t / O_t))`,
and the Yang-Zhang weight

```math
k = 0.34 / (1.34 + (n + 1) / (n - 1))
```

Yang & Zhang (2000). Adds an overnight-gap term missing from RS; for
equity data (where overnight moves matter) this is the estimator with
the lowest variance under the widest range of drift and gap regimes.

`prev_close` may be passed explicitly (aligns cleanly with row 0);
otherwise the previous row's close is used and row 0 is dropped from
the overnight term. The final variance is clamped at zero before the
sqrt to prevent NaN from tiny negatives.

### 6. Positivity guard

Every estimator that computes `log(price / price)` requires strictly
positive prices. A single non-positive value raises `KuantValueError`
with a `KE-VAL-POSITIVE` code. This catches the common mistake of
passing log-returns where prices were expected.

### 7. NaN handling: mean over finite contributions only

Each estimator forms a per-bar contribution array (log-ratios,
Rogers-Satchell terms, overnight variance inputs) and then aggregates
via `mean` or `sum` over the finite mask. Any bar whose OHLC row
contains a NaN is silently excluded from the aggregate. If nothing
remains finite, the return is NaN.

## Return shape

| Kernel | Returns |
| --- | --- |
| `atr` | 1D `np.ndarray` of length `n`; first `window - 1` entries NaN |
| `parkinson` | scalar `float` |
| `garmanklass` | scalar `float` |
| `rogerssatchell` | scalar `float` |
| `yangzhang` | scalar `float` |

The four estimators return a single volatility over the full input
window. For a rolling volatility series, feed each rolling slice
through the estimator, or use `kuant.stats.rollstd` on log returns
as a lightweight sibling.

## Examples

```python
>>> import numpy as np
>>> from kuant.stats import atr, parkinson, yangzhang
>>> rng = np.random.default_rng(0)
>>> close = 100 + np.cumsum(rng.standard_normal(500))
>>> high = close + np.abs(rng.standard_normal(500))
>>> low = close - np.abs(rng.standard_normal(500))
>>> open_ = close + 0.1 * rng.standard_normal(500)
>>> a = atr(high, low, close, window=14)
>>> np.isnan(a[:13]).all()
True
>>> parkinson(high, low) > 0
True
>>> yangzhang(open_, high, low, close) > 0
True
```

## References

- Parkinson, M. (1980). "The Extreme Value Method for Estimating the
  Variance of the Rate of Return." Journal of Business, 53(1),
  61 to 65.
- Garman, M. B., & Klass, M. J. (1980). "On the Estimation of Security
  Price Volatilities from Historical Data." Journal of Business, 53(1),
  67 to 78.
- Rogers, L. C. G., & Satchell, S. E. (1991). "Estimating Variance
  from High, Low, and Closing Prices." Annals of Applied Probability,
  1(4), 504 to 512.
- Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility
  Estimation Based on High, Low, Open, and Close Prices." Journal
  of Business, 73(3), 477 to 491.
- Wilder, J. W. (1978). "New Concepts in Technical Trading Systems."
  Trend Research.

## Related kernels

- `kuant.stats.rollstd`: return-based rolling volatility.
- `kuant.stats.rollema`, `kuant.stats.rollemastd`: EWMA vol siblings.
- `kuant.stats.rollrange`: raw range on any series (not price-level).

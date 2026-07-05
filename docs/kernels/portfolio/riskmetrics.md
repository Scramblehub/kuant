# riskmetrics: Beyond Sharpe and Sortino

## Purpose

Tearsheet-standard risk-adjusted metrics: Omega, Ulcer Index, Kelly,
up/down capture, probabilistic and deflated Sharpe, and a top-N
drawdown episode table.

Every kernel returns either a scalar float or a `dataclass` with a
`.summary()` formatter for human-readable printing.

## Public API

```python
from kuant.portfolio import (
    omega,
    ulcer_index,
    kelly,
    up_capture,
    down_capture,
    probabilistic_sharpe,
    deflated_sharpe,
    drawdown_table,
)

om  = omega(returns, threshold=0.0)                           # -> float
ui  = ulcer_index(equity)                                     # -> UlcerResult
kf  = kelly(returns, cap=1.0)                                 # -> float
uc  = up_capture(returns, benchmark)                          # -> float
dc  = down_capture(returns, benchmark)                        # -> float
psr = probabilistic_sharpe(sharpe, n, skew, kurt, benchmark)  # -> float
dsr = deflated_sharpe(sharpe, n, n_trials, var_sharpes, ...)  # -> float
tbl = drawdown_table(equity, top_n=5)                         # -> DrawdownTableResult
```

## Design decisions

### 1. Omega: the Sharpe alternative for fat tails

`omega = E[max(r - t, 0)] / E[max(t - r, 0)]` over the finite return
distribution above / below threshold `t`. Values above 1 favor the
strategy. Insensitive to Gaussian assumptions.

Edge cases: no downside excursions and at least one upside excursion
returns `+inf`; all-downside returns `0.0`; both zero returns `1.0`
(the "flat" state). All-NaN input raises rather than returns NaN.

### 2. Ulcer Index: pain-path RMS drawdown

Max drawdown reports only the single worst peak-to-trough. The Ulcer
Index takes the RMS of the drawdown series (expressed in percent) and
therefore captures depth AND duration together. UPI (Ulcer Performance
Index) mirrors Sharpe but uses Ulcer as the risk denominator:

```math
UPI = mean(log_returns * 100) / UI
```

Equity input must be strictly positive; convert returns first with
`np.cumprod(1 + returns)`.

### 3. Kelly: continuous-time approximation

`f* = mean(r) / var(r)`, clipped to `[0, cap]`. This is the small-
return / continuous-time approximation. The exact discrete Kelly
requires a per-outcome payoff table and is out of scope; the
approximation is what practitioners actually use for periodic returns.

`cap < 1` implements the standard half-Kelly (or quarter-Kelly) safety
margin. Zero mean or non-finite variance returns 0 without erroring.

### 4. Up / down capture: benchmark-conditional split

`up_capture = mean(r | b > 0) / mean(b | b > 0)`. Symmetric for
`down_capture`. Both align the two 1D series to a common finite mask
before slicing. `CaptureResult` bundles the pair with the ratio
`up/down` and the two conditional sample sizes; individual scalars
are available via the two direct functions.

### 5. Probabilistic Sharpe (PSR): Bailey & Lopez de Prado 2012

Given an observed Sharpe `S`, sample size `n`, return-series skew and
raw kurtosis, PSR is `Phi(z)` where

```math
z = (S - S_bench) * sqrt(n - 1) / sqrt(1 - skew * S + (kurt - 1) / 4 * S^2)
```

Interpretation: the probability that the TRUE Sharpe exceeds
`S_bench`. Kurtosis is raw (Gaussian = 3), NOT excess. Small samples
(`n < 30`) emit a `KuantNumericWarning` because the CDF approximation
degrades.

### 6. Deflated Sharpe (DSR): Bailey & Lopez de Prado 2014

Adjusts PSR for the number of alternative strategies tested. Under
the null of zero true Sharpe, the expected maximum observed Sharpe
across `N` trials is approximately

```math
E[max S] ≈ sqrt(V) * ((1 - gamma) * Phi^{-1}(1 - 1/N)
                    + gamma      * Phi^{-1}(1 - 1/(N * e)))
```

where `gamma` is the Euler-Mascheroni constant and `V` is the
variance of the Sharpes across trials. DSR then calls PSR with this
expected max as the benchmark, so the returned probability is the
chance the observed Sharpe survives selection bias.

### 7. Drawdown table: walk once, merge overlapping episodes

`drawdown_table` walks the equity curve once:

1. Advance until the current bar sits below the running max.
2. Locate the peak by scanning backward from that bar.
3. Track the running trough while advancing forward.
4. Close the episode either when equity reclaims the peak (recovery)
   or when the series ends (still underwater; `recovery` is `None`).
5. Skip past the recovery bar and continue.

Episodes are sorted by `depth` (most negative first) and truncated to
`top_n`. Overlapping arcs are merged by construction: a drawdown does
not open a new episode until the previous peak is reclaimed.

## Return shape / dataclass fields

**UlcerResult**: `ulcer_index: float`, `upi: float`, `n: int`.

**CaptureResult**: `up_capture`, `down_capture`, `up_down: float`,
`n_up: int`, `n_down: int`.

**DrawdownTableResult**: parallel lists of length `n`:

| Field | Type | Meaning |
| --- | --- | --- |
| `peaks` | list[int] | Peak bar index for each episode |
| `troughs` | list[int] | Trough bar index |
| `recoveries` | list[int or None] | Peak-reclaim bar; None if underwater |
| `depths` | list[float] | Trough / peak - 1 (negative) |
| `durations` | list[int] | `trough - peak` in bars |
| `recovery_times` | list[int or None] | `recovery - trough` in bars |
| `n` | int | Episodes returned (up to `top_n`) |

`omega`, `kelly`, `up_capture`, `down_capture`, `probabilistic_sharpe`,
`deflated_sharpe` return `float`.

## Examples

```python
>>> import numpy as np
>>> from kuant.portfolio import omega, probabilistic_sharpe
>>> rng = np.random.default_rng(0)
>>> r = rng.normal(0.001, 0.01, 2520)
>>> omega(r) > 1.0                                        # positive drift
True
>>> probabilistic_sharpe(1.0, 252) > 0.9                  # Sh=1 on 252 obs
True
```

## Related kernels

- `kuant.portfolio.sharperatio`, `kuant.portfolio.sortinoratio`: the
  canonical two ratios.
- `kuant.portfolio.drawdown`: the underlying drawdown series that
  feeds `ulcer_index` and `drawdown_table`.
- `kuant.stats.rollsharpe`, `kuant.stats.rollsortino`: rolling
  windowed versions used in tearsheets.

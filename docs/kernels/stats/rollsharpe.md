# rollsharpe — Rolling Sharpe ratio

## Purpose

Excess-return per unit of volatility over a trailing window, annualized:

```math
\text{rollsharpe}(r, w, \text{ann})[t]
  = \frac{\overline{r}_w - r_f}{\sigma_{r,w}} \cdot \sqrt{\text{ann}}
```

## Public API

```python
from kuant.stats import rollsharpe

s = rollsharpe(x, window, ann_factor=1.0, rf=0.0, ddof=1)
```

- `x` — 1D periodic returns.
- `window` — trailing window size (bars).
- `ann_factor` — annualization scaler. Daily → 252, weekly → 52, monthly → 12.
- `rf` — per-period risk-free rate subtracted from returns.
- `ddof` — sample-std ddof (default 1).

## Design decisions

### Composition of primitives

Built as `(rollmean(x, w) - rf) / rollstd(x, w, ddof)` then annualized.
No new inner loop — reuses the existing rolling primitives.

### NaN semantics

Warm-up (first `window - 1` bars): NaN. Windows containing any NaN input:
NaN. Windows with zero std: NaN (undefined Sharpe).

### Backend-preserving

numpy in → numpy out; cupy in → cupy out. Inherits from `rollmean` /
`rollstd`.

## Related

- `rollsortino` — downside-only variant
- `rollcalmar` — return / |max drawdown|
- `rollmean`, `rollstd` — underlying primitives

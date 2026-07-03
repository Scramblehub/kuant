# rollsortino — Rolling Sortino ratio

## Purpose

"Downside-only Sharpe" — replaces the denominator's total volatility with
downside deviation, penalizing only returns below a target (MAR):

```math
\text{downside dev}(r, w, \tau) = \sqrt{\overline{(\max(\tau - r, 0))^2}}
```

```math
\text{rollsortino}(r, w, \text{ann}, \tau)[t]
  = \frac{\overline{r}_w - \tau}{\text{downside dev}} \cdot \sqrt{\text{ann}}
```

Reduces to Sharpe when returns are symmetric and `target = mean`.

## Public API

```python
from kuant.stats import rollsortino

s = rollsortino(x, window, ann_factor=1.0, target=0.0)
```

- `target` — Minimum Acceptable Return per period. Returns below contribute
  to the downside deviation; returns above do not.

## When to prefer Sortino over Sharpe

- Non-Gaussian returns with asymmetric fat tails (crypto, options strategies)
- Strategies where upside volatility should not be penalized
- Post-hoc portfolio comparison when compounding matters

## Design decisions

### Uses rollmean on squared downside excursions

Computed as `rollmean((target - r).clip(min=0)^2, window)`, then `sqrt()`.
No new inner scan required.

### NaN when no downside in the window

If the whole window has `r >= target`, downside dev is 0 and the ratio
is undefined. Returns NaN. Matches the convention that "no losses
observed" is a meaningless denominator.

## Related

- `rollsharpe`
- `rollcalmar`

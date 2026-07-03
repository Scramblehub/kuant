# rollcalmar — Rolling Calmar ratio

## Purpose

Annualized mean return divided by absolute max drawdown:

```math
\text{rollcalmar}(r, w, \text{ann})[t]
  = \frac{\overline{r}_w \cdot \text{ann}}{|\text{rollmdd}(r, w)[t]|}
```

Calmar rewards strategies that keep drawdowns small relative to return.
Tail-averse position sizing often uses Calmar rather than Sharpe.

## Public API

```python
from kuant.stats import rollcalmar

c = rollcalmar(x, window, ann_factor=1.0)
```

## Design decisions

### Composition of rollmean + rollmdd

Trivial arithmetic on top of the two component kernels. Reuses their
NaN semantics and backend handling.

### NaN when no drawdown

If the window has zero drawdown (monotonically increasing equity),
the ratio is undefined. Returns NaN. Users who want to substitute
a large positive value should do so at the call site.

## Related

- `rollmdd` — the underlying primitive
- `rollmean`
- `rollsharpe` — vol-adjusted; less tail-sensitive

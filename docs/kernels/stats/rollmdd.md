# rollmdd — Rolling maximum drawdown

## Purpose

For each trailing window, build the (local) equity curve and return the
maximum drawdown observed within it:

```
equity[j] = ∏ (1 + r[t-w+1+k])   for k in 0..j
peak[j]   = max(equity[0..j])
dd[j]     = equity[j] / peak[j] - 1
rollmdd[t]= min(dd[j])
```

Returned as a NEGATIVE number (e.g. -0.15 for a 15% drawdown).

## Public API

```python
from kuant.stats import rollmdd

mdd = rollmdd(x, window)
```

## Complexity

`O(n · w)` — the drawdown depends on the entire window's shape and can't
be reduced to a cumsum trick.

## Design decisions

### Local equity curve per window (not global)

Each window starts fresh at equity = 1. A drawdown in an early window
does not propagate into later windows; each is a self-contained history.
This matches how traders think about "recent drawdown over the past N days."

### NaN in window → NaN result

Strict-window semantics: if any bar in the trailing window is NaN,
the result is NaN. No partial computation.

### GPU input transparently handled

Because the outer loop is Python-level, GPU dispatch overhead per
window would dominate for reasonable window sizes. Cupy input is
transferred to CPU, computed with numpy, and transferred back. The
backend-preserving API is maintained.

## Related

- `rollcalmar` — mean return / |rollmdd|
- `rollsharpe` — vol-adjusted variant

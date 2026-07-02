# rollidio — Rolling idiosyncratic (residual) volatility

## Purpose

`rollidio(y, x, w)[i]` = standard deviation of the residuals from the
OLS regression `y = α + β·x + ε` fit over the trailing window.

Answers "how much of `y`'s variability is NOT explained by `x`?"

Direct use in kuant:
- Idiosyncratic vol as a factor signal ("names with high residual vol
  after factoring out market")
- Single-name factor-model residuals for pairs / basket construction
- Detection of regime shifts where the linear relationship breaks

## Public API

```python
from kuant.stats import rollidio
result = rollidio(y, x, window, ddof=1)
```

**Argument order** — `y` first, `x` second. Matches the regression
semantic "y explained by x".

## Design decisions

### Closed-form composition — no per-window residual computation

Fact:

```math
var(ε) = var(y) - cov(x, y)² / var(x)
       = var(y) · (1 - corr(x, y)²)
```

So:

```math
rollidio(y, x, w) = √(rollvar(y) · (1 - rollcorr(x, y)²))
```

Composes on `rollstd` and `rollcorr`. No new cumsum work.

### Guard `1 - corr²` against FP negatives

Floating-point noise can push `corr` slightly above 1 in machine-
perfect-correlation cases, making `1 - corr²` a small negative
number. `xp.maximum(0, 1 - corr²)` clamps to 0 (idio vol is
non-negative by definition).

### Everything else inherited

Backend, dtype, NaN, edge cases all inherit from the composed
kernels.

## Cross-check tests

- Perfectly linear `y = 2x + 3` gives idio ≈ 0 (atol=1e-6 to account
  for FP cancellation near corr=1)
- Independent `y ⊥ x` gives idio ≈ std(y) (median ratio > 0.85 over
  many random windows)
- Closed-form matches manual reconstruction:
  `rollidio == rollstd(y) · √(1 - rollcorr(x,y)²)`

## Test coverage (3 tests)

Perfect correlation, uncorrelated case, closed-form verification.

## Direct usage in kuant

- Rolling idiosyncratic vol of single-name returns vs SPY → factor
  for signal construction
- Regime detection: sudden rise in rollidio suggests the linear
  relationship is breaking down
- Beta-adjusted risk metrics for pairs / basket strategies

## Related kernels

- `kuant.stats.rollstd` — the "unexplained variance" numerator
- `kuant.stats.rollcorr` — the "explained fraction" via 1 - corr²
- `kuant.stats.rollbeta` — sibling regression output (slope)

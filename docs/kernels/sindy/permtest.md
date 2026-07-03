# permtest — Universal permutation null-hypothesis test

## Purpose

Shuffle-the-target permutation test. Give it a metric you computed on
your data, a `metric_fn(x, y_shuffled)` callable, `x` and `y`, and it
runs N permutations returning a p-value.

The gold-standard null test for a null-heavy research pipeline. If your
"signal" performs no better than a random shuffle of the target, it's
noise.

## Public API

```python
from kuant.sindy import permtest

result = permtest(
    real_metric,
    metric_fn,
    x, y,
    n_perms=1000,
    seed=0,
    higher_is_better=True,
)
print(result.summary())          # readable output
result.p_value                    # float
```

## Design decisions

### +1 correction on the p-value

```math
p = (# permuted ≥ real + 1) / (n_perms + 1)
```

Standard convention. Prevents `p = 0` when `n_perms` is small; also
asymptotically correct. `p ≥ 1 / (n_perms + 1)` always.

### higher_is_better flag

Default True (metrics like R², correlation, Sharpe). Flip for
metrics where lower is better (MSE, drawdown, tracking error).

### No dependencies beyond numpy

Pure numpy loop over permutations. Cheap enough for a few thousand
runs. For very expensive `metric_fn`, users should batch smartly
outside of this call.

## Real-world use in our research

Ran hundreds of these on V8 during production research. Without them
we would have shipped multiple garbage signals. Specific saves:

- SINDy #8 PINN-lite: apparent gate looked great (+7.26pp CAGR,
  +0.61 Calmar), but `p_value = 0.50` — half of shuffled runs did as
  well. Killed before shipping.
- Cross-derivative V5: real signal (`p = 0.004`) but not enough Sharpe
  lift; kept as caveat, not shipped.

## Related tools

- `kuant.sindy.grangerscan` — often followed by permtest to confirm
  Bonferroni hits aren't multiple-testing artifacts
- `kuant.qm.belltest` — companion "did we beat classical bounds" test

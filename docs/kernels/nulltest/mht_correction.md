# mht_correction: Multiple-hypothesis testing corrections

## Purpose

When you test `N` strategies (or signals, or backtest configurations)
and pick the best one, the observed significance is inflated by the
selection process. `mht_correction` adjusts a vector of raw p-values
under one of three standard corrections:

- **Bonferroni**: multiply raw p by `N`. Simplest, most conservative.
  Controls family-wise error rate under ANY dependence structure.
- **Holm**: step-down variant of Bonferroni. Uniformly more powerful.
- **BH (Benjamini-Hochberg)**: controls the False Discovery Rate
  instead of the family-wise error rate. Less conservative; standard
  in research settings that tolerate a few false positives.

## Public API

```python
from kuant.nulltest import mht_correction

adj = mht_correction(p_values, method='bh')
```

- `p_values`: 1D array of raw p-values in `[0, 1]`, or a scalar.
- `method`: `'bonferroni'`, `'holm'`, `'bh'`. Default `'bh'`.

Return type mirrors the input: scalar in, scalar out; 1D array in,
1D array out.

## Design decisions

### 1. Bonferroni: multiply and cap

```python
p_adj = min(1.0, N * p_raw)
```

Applied elementwise. Under any dependence structure, controls the
family-wise error rate at exactly `alpha` if you reject any hypothesis
with `p_adj < alpha`. The price is low power when `N` is large: a
p-value of `0.001` on 100 tests becomes `0.1`, no longer significant.

### 2. Holm: sorted step-down with monotone enforcement

Sort p-values ascending. Scale the k-th smallest by `(N - k)`:

```
p_adj_sorted[k] = (N - k) * p_sorted[k]         for k = 0..N-1
```

Enforce non-decreasing monotonicity from left to right via
`np.maximum.accumulate`, then clamp to 1.0 and unsort with the
inverse permutation. Uniformly more powerful than Bonferroni and
retains the same family-wise error control.

### 3. BH: FDR control with top-down monotone enforcement

Sort p-values ascending. Scale by `N / (k + 1)` where `k` is the
1-based rank:

```
p_adj_sorted[k] = p_sorted[k] * N / (k + 1)
```

Enforce non-INCREASING monotonicity from the top down via
`np.minimum.accumulate(scaled[::-1])[::-1]`, clamp to 1.0, and unsort.
The top-down direction is what distinguishes BH from Holm: BH allows
LARGER adjusted p-values earlier in the sort as long as smaller ones
appear later.

BH controls FDR (expected fraction of rejections that are false
discoveries) at `alpha`, which is a strictly weaker guarantee than
family-wise error rate but sufficient for research screens.

### 4. Input validation

- Any p outside `[0, 1]` raises `KuantValueError` with the offending
  index (helps trace the source of a bad p).
- Scalar input is detected via `np.isscalar` before validation; the
  result comes back as a Python float rather than a length-1 array.
- 1D validation via `require_1d` runs on non-scalar inputs only.

### 5. Vectorized throughout

All three methods use `np.argsort` + array arithmetic. No Python
loops. Even for `N = 100000` this returns in milliseconds.

## Return shape

- Scalar input → `float`.
- 1D array input → 1D `np.ndarray` of the same length, dtype float64.

All returned values are in `[0, 1]`.

## Examples

```python
>>> import numpy as np
>>> from kuant.nulltest import mht_correction
>>> raw = np.array([0.001, 0.01, 0.04, 0.20, 0.50])
>>> mht_correction(raw, method='bonferroni').tolist()
[0.005, 0.05, 0.2, 1.0, 1.0]
>>> # BH is less conservative than Bonferroni.
>>> bh = mht_correction(raw, method='bh')
>>> bool(bh[0] <= 0.005)
True
```

## References

- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False
  Discovery Rate: A Practical and Powerful Approach to Multiple
  Testing." Journal of the Royal Statistical Society Series B.
- Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test
  Procedure." Scandinavian Journal of Statistics.

## Related kernels

- `kuant.nulltest.spa_test`, `kuant.nulltest.mcs_test`: joint tests
  that make the correction implicitly when a single test is preferred
  over a bulk adjustment.
- `kuant.nulltest.bootstrap_ic`: produces the raw p-values that
  typically feed this function when a screen tests many signals.
- `kuant.portfolio.deflated_sharpe`: the Sharpe-specific analog of
  a multiple-testing correction.

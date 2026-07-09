# hrp: Hierarchical Risk Parity

## Purpose

Allocate capital across `n` assets without inverting the covariance
matrix. Lopez de Prado 2016 introduced HRP as a robust alternative to
mean-variance optimization: singular or near-singular `Sigma` breaks
Markowitz, but HRP only ever reads variances and pairwise variance
comparisons.

Three steps:

1. Convert correlation to a distance: `dist = sqrt((1 - corr) / 2)`.
2. Single-linkage hierarchical cluster on the distance to get a leaf
   ordering.
3. Recursive bisection down the tree, splitting capital between
   sibling subclusters by inverse-variance weights.

## Public API

```python
from kuant.portfolio import hrp

r = hrp(cov)                    # correlation derived from cov
r = hrp(cov, corr=corr_matrix)  # explicit correlation
r.weights                       # (n,), sum to 1, non-negative
r.order                         # asset indices in clustered order
r.linkage                       # scipy Z matrix
print(r.summary())
```

- `cov` — 2D (n, n) covariance. Any dtype; promoted to float64.
- `corr` — optional (n, n) correlation. Derived from `cov` diagonals
  if omitted.
- Returns `HrpResult(weights, order, linkage)`.

Requires scipy for `scipy.cluster.hierarchy.linkage`. Raises
`KuantValueError [KE-DEP-MISSING]` if scipy is absent.

## Design decisions

### 1. Distance metric: `sqrt((1 - corr) / 2)`

Lopez de Prado's mapping. Bounded in `[0, 1]`, zero when
`corr == 1`, one when `corr == -1`. Diagonal is forced to zero
before clustering so scipy does not choke on FP noise.

### 2. Single linkage on the condensed distance

`scipy.cluster.hierarchy.linkage(condensed, method="single")` where
`condensed = dist[np.triu_indices(n, k=1)]`. Single linkage produces
the "chain" tree that HRP's recursive bisection expects: the leaf
order it induces places correlated assets adjacent to each other, so
splitting the ordered list in half at each recursion respects the
cluster structure.

### 3. Recursive bisection with inverse-variance split

At each level, the ordered index list is halved. The two halves each
form a sub-portfolio weighted by inverse variance (ivp). The
sub-portfolio variances feed the split:

```math
alpha = 1 - var_a / (var_a + var_b)
```

`alpha` goes to the left half, `1 - alpha` to the right. Recurse
into both halves. The final `weights` vector is the product of every
alpha it encountered on its path from root to leaf, then renormalized
to sum to one.

### 4. No matrix inversion, ever

Only `diag(cov)` (through `_ivp_var`) is inverted. Singular full-rank
`Sigma` matrices, common when `n` approaches `T`, degrade Markowitz
into random assignments; HRP produces the same shape of answer under
that regime as under a well-conditioned one.

### 5. Non-negative weights by construction

Every alpha lands in `[0, 1]` (a ratio of non-negative variances),
and initial weights are ones. The product stays non-negative and the
final renormalization preserves sign.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `n == 1` | weight = 1.0, order = [0], no clustering |
| Singular `cov` | works (no inversion); ivp uses `1 / max(diag, 1e-12)` |
| Non-square `cov` | raises `KuantShapeError [KE-SHAPE-2D]` |
| `corr.shape != cov.shape` | raises `KuantShapeError [KE-SHAPE-2D]` |
| scipy not installed | raises `KuantValueError [KE-DEP-MISSING]` |

## Cross-check tests

- `test_weights_sum_to_one` — 10 assets, weights renormalize.
- `test_weights_nonneg` — recursion product stays non-negative.
- `test_bad_shape_rejected` — non-square input.

`tests/portfolio/test_construction_batch6.py::TestHrp`.

## References

- Lopez de Prado 2016, "Building diversified portfolios that
  outperform out-of-sample," Journal of Portfolio Management 42(4).

## Related kernels

- `kuant.portfolio.riskparity` — equal risk contribution, the
  Maillard-Roncalli-Teiletche 2010 counterpart. Both are
  covariance-only allocators; HRP does not invert while riskparity
  does (indirectly through coordinate descent).
- `kuant.portfolio.mintorsion` — diagnostic for effective number of
  bets on the HRP output.

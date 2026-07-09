# pcalgo — PC-algorithm skeleton via Fisher-Z partial correlation

## Purpose

Estimates the undirected causal skeleton of a Bayesian network from
observational data. Given variables `V` and an observed matrix
`X in R^{n x |V|}`, returns an adjacency matrix where `adj[i, j] = 1`
iff the edge `i - j` survives every conditional-independence test up
to order `max_order`.

Skeleton phase only. The full PC algorithm follows the skeleton with
Meek-orientation rules (v-structures then propagation) to produce a
CPDAG; that phase is deferred to a later kuant release. Callers get
the CPDAG PRECURSOR skeleton, which is what most causal-discovery
pipelines consume for feature-set pruning and dependency-structure
analysis.

## Public API

```python
from kuant.causal import pcalgo

res = pcalgo(data, alpha=0.05, max_order=2)
print(res.summary())
print(res.adj)         # (p, p) symmetric adjacency
print(res.sepsets)     # {(i, j): conditioning set that separated them}
```

- `data` : 2D array `(n, p)`. Continuous observed data, one column per
  variable. Assumed jointly Gaussian for the Fisher-Z test statistic
  to hold exactly; robust in practice under mild non-Gaussianity.
- `alpha` : float in `(0, 1)`, default `0.05`. CI-test significance
  level. Higher `alpha` retains more edges (more false positives,
  fewer false negatives).
- `max_order` : positive int, default `3`. Maximum conditioning set
  size. Bounds runtime.

Returns `PcAlgoResult` with `adj`, `sepsets`, `n_ci_tests`,
`max_order_used`, `alpha`, `n`, `p`.

## Design decisions

### 1. Fisher-Z partial correlation as the CI test

Standard choice for Gaussian data (Kalisch-Buhlmann 2007). For `i`,
`j` given conditioning set `S`, compute the partial correlation `rho`
by inverting the covariance of `X[:, S U {i, j}]` and reading the
off-diagonal entry (`_partial_corr`). Fisher-Z transforms `rho` to an
asymptotically standard normal statistic:

```math
z = \tfrac{1}{2} \log \frac{1 + \rho}{1 - \rho}, \quad
\text{stat} = z \sqrt{n - |S| - 3}
```

Two-sided p-value via `erfc`. Reject independence when `p <= alpha`;
keep the edge.

Chosen for exactness under Gaussianity and O(|S|^3) per test cost.
Binary/ordinal data need a chi-square variant which is out of scope
for this kernel.

### 2. Skeleton only, no Meek orientation

The PC algorithm has two phases. Phase 1 (implemented here) removes
edges via CI tests. Phase 2 (Meek rules) orients edges using
v-structure detection and repeated propagation.

Kept phase 1 alone because (a) the skeleton is the object most
downstream consumers actually want (feature pruning, dependency
maps), (b) Meek orientation is sensitive to CI-test errors, so
shipping only the skeleton keeps false-arrow risk out of the API, and
(c) v0.6 scope. Full CPDAG orientation is queued for a later release.

### 3. Textbook termination rule, not "no removal at order K"

A naive termination is: "if no edge was removed at order K, stop." It
misses edges that survive order K but are separable at order K+1 with
a larger conditioning set. The implemented rule (Spirtes-Glymour-
Scheines and Kalisch-Buhlmann formulation) instead tracks the maximum
neighbor count over surviving edges. If `order + 1 > max_neighbors`,
no remaining edge can host a size-`(order + 1)` conditioning set, and
we stop.

Guarantees the skeleton is stable at the reported `max_order_used`.

### 4. Neighbor set uses adjacency to EITHER endpoint

Conditioning-set candidates for testing edge `(i, j)` are variables
adjacent to `i` OR `j` in the current skeleton (excluding `i` and `j`
themselves). This is the standard PC-stable choice and gives
order-independent output within a run.

### 5. `pinv` on the conditioning covariance

`np.linalg.pinv` rather than `inv` on the sub-covariance in
`_partial_corr`. Guards against near-collinear conditioning sets
without raising; the resulting `rho` is clipped to `[-0.999999,
0.999999]` before Fisher-Z, preventing `log((1+rho)/(1-rho))` from
overflowing at boundary values.

### 6. `sepsets` populated on removal, keyed both ways

`sepsets[(i, j)]` and `sepsets[(j, i)]` both point to the tuple that
separated the pair. Downstream orientation phases need the sepset
lookup to be symmetric; populating both keys avoids a `min(i, j)`
convention that callers would otherwise have to know.

## Edge cases

| Condition | Behavior |
| --- | --- |
| Non-2D `data` | raises via `require_2d` |
| Fewer than 30 rows | raises `[KE-VAL-MIN-CLEAN]` |
| Fewer than 2 variables | raises `[KE-VAL-RANGE]` |
| `alpha` not in `(0, 1)` | raises via `require_range` |
| `max_order <= 0` | raises via `require_positive` |
| Perfectly collinear pair | `rho` clipped, Fisher-Z stat saturates, edge likely kept |
| `n - |S| - 3 <= 0` | p-value defaults to `1.0` (fails to reject, keeps edge) |

## Cross-check tests

- `test_pcalgo_recovers_chain_skeleton` : DGP `A -> B -> C`, `D`
  independent. At `max_order = 2`, edges `A-B` and `B-C` survive,
  edge `A-C` is removed via conditioning on `{B}`, and `D` stays
  isolated.
- `test_pcalgo_fork_creates_two_edges` : DGP `A -> B`, `A -> C`. At
  `max_order = 1`, edges `A-B` and `A-C` survive; edge `B-C` is
  removed via conditioning on `{A}` (unshielded collider case).
- `test_pcalgo_alpha_out_of_range` : `alpha = 1.5` raises.
- `test_pcalgo_too_few_variables` : `p = 1` raises.

## Direct usage in kuant

Feature pruning on factor libraries: run `pcalgo` over a set of
candidate signals and drop features whose only neighbors are already
well-explained by others in the graph. Also a diagnostic for
regime-shift research: fit skeletons per regime and compare adjacency
patterns.

Complexity is `O(|V|^(max_order + 2))` worst case. Keep `max_order`
at 2 or 3 for 20+ variables.

## Related kernels

- [`iv`](iv.md) : once the skeleton exposes a candidate instrument,
  `iv` estimates the coefficient with proper SE.
- [`synthcontrol`](synthcontrol.md) : orthogonal identification path
  when experimental variation exists but structural discovery does
  not.

## References

- Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation,
  Prediction, and Search*, 2nd ed. MIT Press. The PC framework and
  the skeleton-then-orient decomposition.
- Kalisch, M., Buhlmann, P. (2007). "Estimating High-Dimensional
  Directed Acyclic Graphs with the PC-Algorithm." *JMLR* 8. Fisher-Z
  variant, consistency proofs, and the textbook termination rule
  used here.
- Meek, C. (1995). "Causal Inference and Causal Explanation with
  Background Knowledge." *UAI*. Orientation rules for the
  post-skeleton phase (not implemented here).

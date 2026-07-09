# kernelpca — Kernel Principal Component Analysis

## Purpose

Nonlinear dimensionality reduction via the kernel trick (Scholkopf
1998). Rather than diagonalizing the covariance of raw features, we
diagonalize the kernel matrix `K` where `K[i,j] = k(x_i, x_j)` for a
chosen kernel `k`. The leading eigenvectors give a nonlinear embedding.

When raw PCA misses structure because the interesting geometry lives
on a curved manifold (e.g. return-embedding clusters that a linear
projection blurs together), an RBF-kernel PCA can recover it.

Common uses:

- Regime detection on nonlinear return-embedding spaces.
- Denoising with RBF plus inverse transform (sklearn side).
- Feature engineering for downstream models that expect an
  approximately linear feature geometry.

Thin wrapper over scikit-learn's `KernelPCA`.

## Public API

```python
from kuant.signals import kernelpca

result = kernelpca(X, n_components=3, kernel='rbf', gamma=None, degree=3)
result.components              # (n, k) transformed data
result.eigenvalues             # top-k kernel eigenvalues
result.kernel                  # kernel name used
```

- `X`: 2D array, shape `(n, d)`.
- `n_components`: int in `[1, min(n, d)]`. Default 3.
- `kernel`: one of `'rbf'` (default), `'poly'`, `'sigmoid'`,
  `'cosine'`, `'linear'`.
- `gamma`: kernel width. `None` uses sklearn's default `1 / d`.
- `degree`: polynomial degree, applies only to `'poly'`.

Requires scikit-learn; missing dependency raises `KuantValueError`
with `[KE-DEP-MISSING]`.

## Design decisions

### 1. `eigen_solver='dense'` pinned

`KernelPCA` supports `dense`, `arpack`, and `randomized` solvers.
For the typical kuant workload (a few hundred to a few thousand rows,
a handful of components), dense is fastest and deterministic. Pinning
it removes a source of cross-version drift.

For very large `n`, users who need arpack should call sklearn
directly; wrapping every solver variant is not this kernel's job.

### 2. Kernel whitelist mirrors sklearn

Accept exactly the five kernels sklearn's `KernelPCA` supports. Any
other string raises `KuantValueError` with `[KE-VAL-RANGE]` and the
full list. Prevents silent typos falling through to sklearn's less
targeted error.

### 3. `n_components` bounded at `min(n, d)`

The kernel matrix is `(n, n)`, so at most `n` eigenvectors exist; the
raw dimension `d` also caps the meaningful count for linear-family
kernels. Bounded at `min(n, d)`; out-of-range raises `[KE-VAL-RANGE]`.

### 4. Lazy sklearn import

Import happens after shape and range validation. Missing sklearn maps
to `[KE-DEP-MISSING]` with the pip install hint.

## Return shape

**KernelPcaResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `components` | 2D array `(n, k)` | Nonlinear embedding of `X` |
| `eigenvalues` | 1D array | Top-`k` kernel eigenvalues, descending |
| `kernel` | str | Kernel used |

## Edge cases

| Condition | Behavior |
| --- | --- |
| `X.ndim != 2` | `KuantShapeError` `[KE-SHAPE-EXPECTED]` |
| `n_components` out of `[1, min(n, d)]` | `KuantValueError` `[KE-VAL-RANGE]` |
| `kernel` not in whitelist | `KuantValueError` `[KE-VAL-RANGE]` |
| `degree <= 0` | `KuantValueError` via `require_positive` |
| scikit-learn missing | `KuantValueError` `[KE-DEP-MISSING]` |

## Related kernels

- `kuant.signals.whitening`: linear whitening; the pre-processing
  step for linear PCA.
- `kuant.signals.ica`: linear but higher-order independence, not just
  decorrelation.
- `kuant.signals.wavelet`, `kuant.signals.emd`: single-series
  decompositions.

## References

- Scholkopf, B., Smola, A. J., & Muller, K.-R. (1998). Nonlinear
  component analysis as a kernel eigenvalue problem. *Neural
  Computation*, 10(5), 1299-1319.

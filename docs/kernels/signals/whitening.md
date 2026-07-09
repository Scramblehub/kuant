# whitening — ZCA and PCA whitening

## Purpose

Linear transform that decorrelates the columns of a data matrix and
scales them to unit variance. Given `X` of shape `(n, d)`, produce
`X_white = (X - mean) @ W` with `cov(X_white) approximately I`.

Two rotations:

- **PCA whitening**: `W = V @ diag(1/sqrt(lambda))`. Rotates into the
  principal-component basis then scales. Columns of `X_white` are the
  standardized PCs.
- **ZCA whitening**: `W = V @ diag(1/sqrt(lambda)) @ V.T`. Adds the
  reverse rotation so columns of `X_white` stay aligned with the
  original features. Best "closest to identity in Frobenius norm"
  whitening (Kessy-Lewin-Strimmer 2015); the classical choice for ICA
  preconditioning in Bell-Sejnowski 1997 style pipelines.

Uses:

- ICA / kernel-PCA preprocessing.
- Removing scale and correlation artefacts before a downstream model
  that assumes iid inputs.

## Public API

```python
from kuant.signals import whitening

result = whitening(X, method='zca', ridge=1e-6)
result.X_white                 # (n, d) whitened data
result.W                       # (d, d) whitening matrix
result.eigenvalues             # ridge-clipped covariance eigenvalues
result.mean                    # column mean subtracted
```

- `X`: 2D array, shape `(n, d)`. Requires `n >= d + 5` for a
  well-conditioned covariance.
- `method`: `'zca'` (default) or `'pca'`.
- `ridge`: positive float. Added to eigenvalues before inversion to
  handle rank-deficient covariance.

## Design decisions

### 1. `eigh` on the sample covariance, not SVD on `X`

`C = (X - mean).T @ (X - mean) / n` is `(d, d)` and symmetric.
`np.linalg.eigh` is stable and fast for that shape. SVD on `X` would
work too but returns more than we need at higher cost when `n >> d`.

### 2. Ridge clip on eigenvalues before inversion

`eigvals = np.clip(eigvals, ridge, None)`. Rank-deficient inputs (a
constant column, an exact linear dependence between two columns) drive
the smallest eigenvalue to zero; unclipped, `1/sqrt(0)` blows up. The
ridge floor keeps the transform finite and turns the affected
directions into pass-through (they contribute noise near their ridge
scale instead of infinities).

Default `ridge=1e-6` is small enough to leave healthy directions
untouched; users who need heavier regularization pass a larger value.

### 3. ZCA default

ZCA preserves feature interpretability: column `k` of `X_white` is
still "roughly the same variable as column `k` of `X`", only
decorrelated and rescaled. PCA whitening rotates that away. Most
downstream pipelines (ICA, isotropic-noise models) work with either;
ZCA is the safer default when the user might want to inspect
individual columns.

### 4. Minimum `n >= d + 5`

Fewer rows than columns gives a singular sample covariance regardless
of ridge; even `n = d` is knife-edge. The `+ 5` buffer keeps the
condition number from exploding at the boundary. Below the threshold,
raises `KuantValueError` with `[KE-VAL-MIN-CLEAN]`.

### 5. Whitening matrix returned for out-of-sample application

`W` and `mean` are both on the result, so a user can apply the same
transform to a held-out batch: `X_test_white = (X_test - mean) @ W`.

## Return shape

**WhiteningResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `X_white` | 2D array `(n, d)` | Whitened data |
| `W` | 2D array `(d, d)` | Whitening matrix |
| `eigenvalues` | 1D array, len `d` | Ridge-clipped covariance eigenvalues |
| `mean` | 1D array, len `d` | Column mean subtracted before whitening |
| `method` | str | `'zca'` or `'pca'` |

## Edge cases

| Condition | Behavior |
| --- | --- |
| `X.ndim != 2` | `KuantShapeError` `[KE-SHAPE-EXPECTED]` |
| `method` not in `{'zca', 'pca'}` | `KuantValueError` `[KE-VAL-RANGE]` |
| `ridge <= 0` | `KuantValueError` via `require_positive` |
| `n < d + 5` | `KuantValueError` `[KE-VAL-MIN-CLEAN]` |
| Rank-deficient `X` | Eigenvalues clipped to `ridge`; transform stays finite |

## Related kernels

- `kuant.signals.ica`: standard downstream consumer; FastICA needs
  whitened input.
- `kuant.signals.kernelpca`: nonlinear alternative when linear
  decorrelation is insufficient.
- `kuant.signals.neutralize`: single-signal analogue (residual against
  factor exposures rather than full covariance whitening).

## References

- Bell, A. J., & Sejnowski, T. J. (1997). The "independent components"
  of natural scenes are edge filters. *Vision Research*, 37(23),
  3327-3338.
- Kessy, A., Lewin, A., & Strimmer, K. (2015). Optimal whitening and
  decorrelation. *The American Statistician*, 72(4), 309-314.

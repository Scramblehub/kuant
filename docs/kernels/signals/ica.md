# ica — Independent Component Analysis (FastICA)

## Purpose

Recover statistically independent latent sources from a multivariate
observation matrix. Given `X` of shape `(n, d)`, ICA estimates an
unmixing matrix `W` such that `S = (X - mean) @ W.T` has maximally
non-Gaussian, mutually independent columns.

Contrast with PCA: PCA finds directions of maximal variance
(second-order decorrelation). ICA goes further and finds directions
of maximal statistical independence (higher-order structure).

Financial uses:

- Decompose a factor-return matrix into independent latent drivers.
- Blind separation of overlapping alpha sources.
- Preprocess for downstream regime / clustering pipelines that assume
  independent inputs.

Thin wrapper over scikit-learn's FastICA (Hyvarinen-Oja 2000).

## Public API

```python
from kuant.signals import ica

result = ica(X, n_components=5, max_iter=300, tol=1e-4, random_state=0)
result.sources                 # (n, k) recovered independent sources
result.mixing                  # (d, k) estimated mixing matrix
result.unmixing                # (k, d) unmixing matrix
result.converged               # bool
```

- `X`: 2D array, shape `(n, d)`.
- `n_components`: int in `[1, d]`. Default `d` (square unmixing).
- `max_iter`, `tol`: FastICA fixed-point convergence parameters.
- `random_state`: seed for the initial mixing guess. Reproducibility.

Requires scikit-learn; missing dependency raises `KuantValueError`
with `[KE-DEP-MISSING]`.

## Design decisions

### 1. `whiten='unit-variance'` fixed internally

FastICA needs whitened input. Rather than expose the sklearn 0.23+
option and risk silently changing behavior across sklearn versions,
we pin `whiten='unit-variance'`. Users who want a different whitening
should call `kuant.signals.whitening` upstream and pass the whitened
matrix in.

### 2. Convergence surfaced as a warning, not an error

FastICA occasionally stalls near a saddle. Rather than raising and
forcing the caller to catch, we return the (possibly unreliable)
result and emit `KuantNumericWarning` with `[KW-CONV-MAXITER]`. The
`converged` field on the result carries the truth for programmatic
checks.

`converged` is defined as `0 < n_iters < max_iter`. `n_iters == -1`
(sklearn didn't populate `n_iter_`) counts as non-converged.

### 3. Mean returned for reconstruction

`result.mean` is the column mean subtracted before whitening. Needed
to invert the transform: `X_recovered = sources @ mixing.T + mean`.

### 4. Lazy sklearn import

Import happens inside the call, after shape and range validation, so
users who never invoke ICA don't pay import cost or crash on a missing
optional dep. Missing sklearn maps to `[KE-DEP-MISSING]` with the pip
install hint.

## Return shape

**IcaResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `sources` | 2D array `(n, k)` | Recovered independent components |
| `mixing` | 2D array `(d, k)` | Estimated mixing matrix `A` |
| `unmixing` | 2D array `(k, d)` | `W = A^-1` (pseudo-inverse if `k < d`) |
| `mean` | 1D array, len `d` | Column mean removed pre-whitening |
| `n_iters` | int | FastICA iterations used (-1 if not exposed) |
| `converged` | bool | True iff `0 < n_iters < max_iter` |

## Edge cases

| Condition | Behavior |
| --- | --- |
| `X.ndim != 2` | `KuantShapeError` `[KE-SHAPE-EXPECTED]` |
| `n_components` out of `[1, d]` | `KuantValueError` `[KE-VAL-RANGE]` |
| `max_iter <= 0` or `tol <= 0` | `KuantValueError` via `require_positive` |
| scikit-learn missing | `KuantValueError` `[KE-DEP-MISSING]` |
| Non-convergence | `KuantNumericWarning` `[KW-CONV-MAXITER]`, `converged=False` |

## Related kernels

- `kuant.signals.whitening`: preprocessing (PCA / ZCA) if you want an
  explicit whitening step before ICA.
- `kuant.signals.kernelpca`: nonlinear alternative when independence
  is expected in a feature space rather than the raw coordinates.
- `kuant.signals.emd`: single-series (rather than multivariate)
  data-driven decomposition.

## References

- Hyvarinen, A., & Oja, E. (2000). Independent component analysis:
  algorithms and applications. *Neural Networks*, 13(4-5), 411-430.

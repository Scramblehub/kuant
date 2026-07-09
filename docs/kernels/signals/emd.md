# emd — Empirical Mode Decomposition

## Purpose

Decompose a 1D non-stationary signal into intrinsic mode functions
(IMFs) plus a residual, via iterative sifting (Huang 1998). Unlike
wavelets, EMD does not require choosing a basis: the decomposition is
data-driven and adapts to the local frequency content of the signal.

Each IMF is a zero-mean oscillatory component; the residual carries
the long-run trend. Sum of IMFs plus residual reproduces the input
(up to sifting truncation).

## Public API

```python
from kuant.signals import emd

result = emd(x, max_imfs=8, sifting_iters=10)
result.imfs                    # list of 1D arrays, high freq first
result.residual                # long-run trend component
result.n_imfs                  # len(result.imfs)
```

- `x`: 1D array. Non-finite values dropped before decomposition.
- `max_imfs`: positive int, cap on the number of IMFs extracted.
- `sifting_iters`: positive int, sifting passes per IMF (fixed-count
  variant; see design note 2).

Requires `scipy.interpolate.CubicSpline`.

## Design decisions

### 1. Fixed-iteration sifting, not adaptive stopping

Standard EMD uses a stopping criterion (typically Cauchy-style or SD
threshold on successive sifts). This MVP uses a fixed count of sifting
passes per IMF for predictability and speed. Trade-off: a value of 10
is a common default that balances mode-mixing against over-sifting.

Adaptive stopping, EEMD, and CEEMDAN are deferred to a later release.

### 2. Cubic-spline envelopes with endpoint stabilization

Per-sifting-pass, upper and lower envelopes are cubic splines through
the local maxima and minima. Endpoints (`x[0]`, `x[-1]`) are appended
to both extremum lists so the spline does not diverge at the
boundaries. The envelope mean is subtracted from the working signal.

### 3. Reconstruction identity

`sum(imfs) + residual = x_clean` exactly (before floating drift). No
scaling or normalization is applied to the IMFs, so the identity
holds to numerical precision.

### 4. Stopping conditions on the outer loop

The outer loop terminates when any of:

- `max_imfs` reached;
- a candidate has fewer than 2 local maxima or minima (cannot form
  envelopes), signalled by `_sift_once` returning `None`;
- the residual after subtraction becomes monotone (fewer than 2
  extrema left).

### 5. Minimum 64 finite samples

Below 64, envelope splines are dominated by endpoint effects and the
IMF count collapses to 0-1. Raises `KuantValueError` with
`[KE-VAL-MIN-CLEAN]`.

### 6. Lazy scipy import

Scipy is only needed at sift time. Import is deferred until inside
`_sift_once`; missing scipy raises `KuantValueError` with
`[KE-DEP-MISSING]`.

## Return shape

**EmdResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `imfs` | list of 1D arrays | Intrinsic mode functions, finest first |
| `residual` | 1D array | Trend residual: `x_clean - sum(imfs)` |
| `n_imfs` | int | Number of IMFs extracted (may be `< max_imfs`) |

## Edge cases

| Condition | Behavior |
| --- | --- |
| `n < 64` after NaN drop | `KuantValueError` `[KE-VAL-MIN-CLEAN]` |
| `max_imfs <= 0` or `sifting_iters <= 0` | `KuantValueError` via `require_positive` |
| scipy missing | `KuantValueError` `[KE-DEP-MISSING]` |
| Monotone input | `n_imfs == 0`, `residual == x_clean` |
| Sifting yields fewer than 2 extrema | Loop halts early, fewer IMFs than `max_imfs` |

## Related kernels

- `kuant.signals.wavelet`: fixed-basis alternative (Haar / Daubechies).
- `kuant.stats.rollstd`: pair with per-IMF variance for local
  band-power features.

## References

- Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
  Q., Yen, N.-C., Tung, C. C., & Liu, H. H. (1998). The empirical mode
  decomposition and the Hilbert spectrum for nonlinear and
  non-stationary time series analysis. *Proceedings of the Royal
  Society A*, 454(1971), 903-995.

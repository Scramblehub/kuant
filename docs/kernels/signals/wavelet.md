# wavelet — Discrete wavelet decomposition

## Purpose

Multi-level discrete wavelet transform. Splits a 1D signal into a
coarse approximation plus detail coefficients at each dyadic scale.

Uses:

- Trend / noise separation at chosen frequency bands.
- Per-scale variance as features for a predictive model.
- Denoising via soft-threshold on detail coefficients (external step).

Native Haar (Daubechies 1988 db1) runs without dependencies. Higher-
order Daubechies families (`db2`, `db4`) delegate to PyWavelets.

## Public API

```python
from kuant.signals import wavelet

result = wavelet(x, n_scales=5, kernel='haar')
result.approximation           # coarse (low-freq) coefficients
result.details                 # list of detail arrays, finest first
result.variances               # per-scale detail variance
```

- `x`: 1D array. Non-finite values dropped before decomposition.
- `n_scales`: positive int, decomposition depth. Capped internally at
  `floor(log2(len(x_clean)))`.
- `kernel`: `'haar'` (default, native), `'db2'`, `'db4'` (PyWavelets).

## Design decisions

### 1. Native Haar via lifting-style pair sums

Each level halves the input:

```
a[k] = (x[2k] + x[2k+1]) / sqrt(2)   # low-pass (approximation)
d[k] = (x[2k] - x[2k+1]) / sqrt(2)   # high-pass (detail)
```

Odd-length inputs drop the trailing sample. The next level runs on
`a`. Repeat `n_scales` times or until fewer than 2 samples remain.

Orthonormal scaling (`1/sqrt(2)`) preserves the L2 energy identity:
`||x||^2 = ||approx||^2 + sum_k ||details[k]||^2` (up to boundary
truncation), which is the reconstruction identity Mallat 1989
formalised for the pyramid algorithm.

### 2. Daubechies via PyWavelets, lazy import

`db2` and `db4` require PyWavelets. Import is deferred until the
kernel is actually requested. Missing dependency raises
`KuantValueError` with `[KE-DEP-MISSING]` and the pip hint. Native
Haar users pay no import cost.

### 3. NaN pre-filter, not NaN propagation

`arr = arr[np.isfinite(arr)]` strips non-finite entries up front.
Wavelet transforms have no meaningful NaN semantics (any NaN pollutes
its whole subtree). Pre-filtering keeps the recursion clean; length
loss is user-visible via `result.details[k].size`.

### 4. Minimum 32 finite samples

Below 32, the pyramid collapses within one or two levels and the
per-scale variances are dominated by boundary effects. Raises
`KuantValueError` with `[KE-VAL-MIN-CLEAN]`.

### 5. Scale cap at `floor(log2(n))`

Requesting more levels than the dyadic depth silently clamps rather
than raising. The final `n_scales` field reflects what was actually
computed, not what was asked for.

## Return shape

**WaveletResult**

| Field | Type | Meaning |
| --- | --- | --- |
| `approximation` | 1D array | Coarsest low-frequency coefficients |
| `details` | list of 1D arrays | Detail coefficients, finest scale first |
| `variances` | 1D array, len `n_scales` | Sample variance of each detail scale |
| `n_scales` | int | Levels actually computed |
| `kernel` | str | Kernel used |

## Edge cases

| Condition | Behavior |
| --- | --- |
| `n < 32` after NaN drop | `KuantValueError` `[KE-VAL-MIN-CLEAN]` |
| `n_scales <= 0` | `KuantValueError` via `require_positive` |
| `n_scales > log2(n)` | Clamped to `floor(log2(n))` |
| `kernel` not in supported set | `KuantValueError` `[KE-VAL-RANGE]` |
| `kernel` in `{db2, db4}` without pywt | `KuantValueError` `[KE-DEP-MISSING]` |
| Odd-length input at any level | Trailing sample dropped |

## Test coverage

Reference tests: `tests/signals/test_signalproc_batch7.py`. Cover
reconstruction identity, per-scale variance decomposition, dependency
error surfacing, and boundary trimming.

## Related kernels

- `kuant.signals.emd`: adaptive, data-driven analogue (no fixed basis).
- `kuant.signals.kernelpca`: nonlinear feature-space decomposition.
- `kuant.stats.rollstd`: pair with `result.variances` for rolling
  band-power features.

## References

- Daubechies, I. (1988). Orthonormal bases of compactly supported
  wavelets. *Communications on Pure and Applied Mathematics*, 41(7).
- Mallat, S. (1989). A theory for multiresolution signal
  decomposition: the wavelet representation. *IEEE Transactions on
  Pattern Analysis and Machine Intelligence*, 11(7).

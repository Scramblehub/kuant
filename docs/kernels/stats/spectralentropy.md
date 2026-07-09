# spectralentropy — Shannon entropy of the power spectrum

## Purpose

Frequency-domain complexity metric (Inouye 1991). Computes the
Shannon entropy of the normalized periodogram and reports it both in
nats and normalized to `[0, 1]`:

- Normalized ~ 1 -> broadband, near-uniform spectrum (white noise).
- Normalized ~ 0 -> spectral mass concentrated in one bin (pure
  sinusoid).

Complements time-domain complexity measures (Lyapunov exponent,
sample entropy) with a single-number frequency-domain view.

## Public API

```python
from kuant.stats import spectralentropy

r = spectralentropy(x, detrend=True)
```

- `x` — 1D array. Non-finite dropped. Requires `n >= 32`.
- `detrend` — if `True` (default), subtract the sample mean before FFT.
- Returns `SpectralEntropyResult(entropy, normalized, n_bins, n_samples)`.
  - `entropy`: nats. Bounded in `[0, log(n_bins)]`.
  - `normalized`: `entropy / log(n_bins)`. Bounded in `[0, 1]`.
  - `n_bins`: number of positive-frequency bins used (DC dropped),
    equal to `n // 2` for even `n`.

## Design decisions

### 1. `rfft` on real input, DC dropped

Real FFT (`np.fft.rfft`) exploits the conjugate symmetry so we compute
only `n // 2 + 1` bins. The DC bin (`index 0`) is dropped because it
encodes the mean, which the caller controls with `detrend`. Keeping DC
would either double-count the level (if `detrend=False`) or introduce
a zero bin (if `detrend=True`) that biases the entropy downward.

### 2. Normalize power to a probability

```
p_j = power_j / sum(power)
ent = -sum(p_j * log(p_j))       # over p_j > 0
max_ent = log(n_bins)
normalized = ent / max_ent
```

Zero-power bins are dropped (`p[p > 0]`) so `0 * log(0)` never appears.
Skipping vs adding a `+eps` avoids introducing a small floor that
would drift the reported entropy on sparse spectra.

### 3. Total-power collapse guard

If `total < 1e-15` (constant series or all-zero after detrend), returns
`entropy = NaN`, `normalized = NaN`. Downstream sees a NaN rather than
a divide-by-zero warning.

### 4. `n >= 32` floor

Below 32 samples the periodogram has fewer than 16 usable bins and
the normalized entropy loses resolution:

```
KE-VAL-MIN-CLEAN: "only {n} finite values; need at least 32."
```

### 5. Optional `detrend` (default True)

Time series with drift concentrate power in low frequencies purely
because of the trend. Subtracting the mean removes the DC contribution
but does NOT remove linear drift; if drift is present the caller should
difference or fit-and-subtract before calling. The design choice is to
keep the kernel narrow (mean-detrend only); a stronger detrend belongs
to the caller.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x.ndim != 1` | raises `KuantShapeError` `KE-SHAPE-1D` |
| `n < 32` finite | raises `KuantValueError` `KE-VAL-MIN-CLEAN` |
| Constant series | `total < 1e-15`; returns NaN entropy |
| Pure sinusoid | normalized entropy near 0 (concentrated) |
| White noise | normalized entropy near 1 (uniform) |
| Linear ramp with `detrend=True` | trend remains; entropy low |
| `detrend=False` on non-zero-mean input | DC dropped, so mean does not enter the entropy directly |

## Cross-check tests

- `test_white_noise_near_one` — 1024-point Gaussian: `normalized > 0.85`
- `test_sinusoid_near_zero` — pure sinusoid period 32: `normalized < 0.20`
- `test_too_short_rejected` — 20-point input raises `KuantValueError`

## References

- Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S.,
  Iyama, A., Katsuda, Y., Hirano, M. (1991). "Quantification of EEG
  irregularity by use of the entropy of the power spectrum."
  Electroencephalography and Clinical Neurophysiology 79, 204-210.
- Shannon, C. E. (1948). "A mathematical theory of communication."
  Bell System Technical Journal 27, 379-423.

## Related

- `kuant.stats.localwhittle` — parametric low-frequency spectrum
- `kuant.stats.mfdfa` — time-domain scaling complement
- `kuant.stats.bdstest` — nonlinear iid test; different lens on the
  same "is this white noise" question

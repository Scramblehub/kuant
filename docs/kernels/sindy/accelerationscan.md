# accelerationscan — Second-derivative predictive-power scan

## Purpose

For a series `x` with target `y`, compute smoothed second derivatives
of `x` at multiple bandwidths and report the correlation of each
acceleration variant with `y`. Tests the physics analog: does knowing
acceleration predict the target?

Small and specific. If your target is `x` itself at a future lag
(e.g. `y = x.shift(-h)`), this becomes a self-predictability test.
For most daily-frequency financial series, the answer is no.

## Public API

```python
from kuant.sindy import accelerationscan

result = accelerationscan(
    x, target,
    smoothings=[1, 5, 21, 63],
    noise_floor=0.025,
)
print(result.summary())
```

Returns `AccelerationScanResult` with per-smoothing correlations,
sample counts, peak smoothing, peak correlation, and the noise-floor
threshold.

## Design decisions

### Discrete second difference + centered MA

Second derivative approximated by `d²x[t] = x[t] - 2·x[t-1] + x[t-2]`.
Then a centered moving-average of length `smoothing` smooths the
result. NaN at boundaries where the smoothing window isn't fully
populated.

For `smoothing = 1`, no averaging is applied — raw `d²x`.

### Noise-floor threshold

Default `0.025`. Rule of thumb: below this, sample correlations on
typical daily-frequency financial series are indistinguishable from
random. Users should scale up with `1/√n` for shorter samples:

- `n = 500`: sample corr std ~ 0.045 → noise_floor 0.05 is safer
- `n = 5000`: sample corr std ~ 0.014 → noise_floor 0.025 is fine

If `abs(peak_corr) < noise_floor`, the summary tags a null result and
explains that returns are martingale-like at this frequency.

### No permutation test — kept lightweight

Unlike `pinnscan`, this tool doesn't run a permutation null. If you
find a candidate above the noise floor, follow up with
`permtest(peak_corr, ...)` explicitly.

## Canonical null pattern

The typical daily-frequency financial-returns result: all tested
smoothings give `|corr| < 0.025` — within the sample-size-dependent
noise floor. Clean null. Returns are effectively martingale at daily
frequency at that scale, and the summary auto-diagnostic tag
surfaces the null verdict.

## Related tools

- `kuant.sindy.permtest` — chain after this to confirm any above-
  noise-floor findings with a permutation null
- `kuant.stats.rollmean` — the MA under the hood (though this kernel
  uses a direct implementation for the centered convention)

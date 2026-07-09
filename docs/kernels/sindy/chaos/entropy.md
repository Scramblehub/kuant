# entropy: permutation, sample, approximate, dispersion, transfer

## Purpose

Five nonlinear complexity measures that quantify how "structured" a
time series is from different angles:

- `permutationentropy` (Bandt-Pompe 2002): Shannon entropy of the
  ordinal-pattern distribution over delay-embedded windows. Fast,
  robust to any monotone transform of the signal, and the default
  first-pass complexity screen.
- `sampleentropy` (Richman-Moorman 2000): negative log conditional
  probability that two windows close at length `m` remain close at
  length `m + 1`. Removes the self-match bias of ApEn.
- `approximateentropy` (Pincus 1991): the original nonlinear complexity
  measure. Retained for legacy comparison; SampEn is the recommended
  successor.
- `dispersionentropy` (Rostaghi-Azami 2016): Shannon entropy over
  class-symbol patterns obtained by normal-CDF binning. More stable
  than SampEn at short lengths (a few hundred points).
- `transferentropy` (Schreiber 2000): directed information flow
  `X -> Y` via histogram binning of the aligned triples
  `(Y_t, X_t, Y_{t+lag})`. Model-free measure of asymmetric coupling.

All entropies are returned in nats. Small-sample biases where known
are flagged in the individual sections.

## Public API

```python
from kuant.sindy.chaos import (
    permutationentropy, sampleentropy, approximateentropy,
    dispersionentropy, transferentropy,
)

pe = permutationentropy(x, m=3, tau=1)
pe.normalized                    # 0 = monotone, 1 = fully random

se = sampleentropy(x, m=2)       # r auto = 0.2 * std(x)
ae = approximateentropy(x, m=2)
de = dispersionentropy(x, m=3, c=6)

te = transferentropy(x, y, lag=1, bins=6)
te.te                            # nats, X -> Y
```

Return dataclasses: `PermutationEntropyResult`,
`SampleEntropyResult`, `ApproximateEntropyResult`,
`DispersionEntropyResult`, `TransferEntropyResult`, each with a
`.summary()` renderer.

## Design decisions

### `permutationentropy`: ordinal encoding via stable argsort

Rows of the `(m, tau)` embedding are encoded as tuples of their
`np.argsort(..., kind="stable")` output. Stable sort ensures ties
map to a canonical pattern rather than a random one. `m` is bounded
to `[2, 10]`; `m! = 3.6M` at `m = 10` sets a practical ceiling.
`normalized = entropy / log(m!)` scales the result to `[0, 1]`.

### `sampleentropy`: Chebyshev tolerance, no self-match

The internal `_count_matches` walks templates of length `m` and `m + 1`
and counts pairs `(i, j)` with `j > i` whose Chebyshev distance is
under `r`. Excluding `j == i` is the defining move that separates
SampEn from ApEn.

`r` defaults to `0.2 * std(x, ddof=1)`, the Richman-Moorman heuristic.
If either count `A` or `B` is zero, SampEn is undefined and the result
carries `entropy = nan` (soft failure rather than exception, so callers
can screen many series in a loop).

Minimum length: 50 finite values, from `KE-VAL-MIN-CLEAN`.

### `approximateentropy`: retained for legacy, self-match bias documented

The `phi(m)` function INCLUDES self-matches (`templates[i]` matched
against itself), giving ApEn its known bias toward regularity on
short series. `SampleEntropyResult` docstring points callers to
`sampleentropy` when unbiased comparison is needed.

### `dispersionentropy`: normal-CDF class assignment

Values are z-scored and passed through the standard normal CDF, then
scaled by `c` and floored to integers in `[1, c]`. The class sequence
is delay-embedded to shape `(N - (m - 1) * tau, m)` and Shannon
entropy is taken over the observed dispersion patterns. Max entropy
is `log(c ** m)`, used for the `normalized` field.

A near-constant input (`std < 1e-15`) emits
`KW-DE-CONSTANT-INPUT` and returns zero entropy: the measure is not
informative on a flat signal, and callers should be warned rather
than fed a spurious tiny number.

### `transferentropy`: quantile bins, three-marginal ratio

Each of `Y_t`, `X_t`, `Y_{t+lag}` is binned to `bins` classes using
its own quantile edges (endpoints set to `+-inf` so out-of-range
values fall in the extreme bins). Joint counts fill a
`(bins, bins, bins)` tensor and TE is computed via the standard
three-marginal identity:

```
TE(X -> Y) = sum p(y_next, x_t, y_t) * log[
    p(y_next, x_t, y_t) * p(y_t)
    / (p(y_t, x_t) * p(y_t, y_next))
]
```

Cells with zero joint or zero denominator are masked out with the
`np.errstate(divide="ignore", invalid="ignore")` guard.

Histogram TE has a known positive small-sample bias. The docstring
recommends subtracting a shuffled-`y` baseline as a permutation null
when inference at short samples matters.

## Error codes

- `KE-VAL-MIN-CLEAN`:
  - `permutationentropy` needs `m * tau + 1` finite values.
  - `sampleentropy`, `approximateentropy`: 50 finite values.
  - `dispersionentropy`: `m * tau + 1` finite values.
  - `transferentropy`: `100 + lag` paired finite values.
- `KE-SHAPE-EQUAL-LEN`: `transferentropy` inputs of unequal length.
- `KW-DE-CONSTANT-INPUT`: `dispersionentropy` on a near-constant series.
- Standard range / positivity errors from `_validation` for `m`,
  `tau`, `c`, `bins`, `r`, `lag`.

## Edge cases

| Condition | Behavior |
| --- | --- |
| Strictly monotone `x` | `permutationentropy.entropy = 0`, `n_patterns_seen = 1`. |
| Gaussian noise, `m = 3` | `permutationentropy.normalized > 0.95`. |
| Sinusoid vs Gaussian | `sampleentropy(sin) < sampleentropy(noise)`. |
| Zero pattern matches | `SampleEntropyResult.entropy = nan` (soft). |
| Constant input to `dispersionentropy` | `KW-DE-CONSTANT-INPUT`, `entropy = 0`. |
| Independent `x`, `y` | `transferentropy.te` small but positive (bias). |
| `y[t+1] = 0.7 * x[t] + noise` | `TE(x -> y) > TE(y -> x)`. |
| Fewer finite values than the kernel-specific floor | `KE-VAL-MIN-CLEAN`. |

## When it fires

- Regime classification: `permutationentropy.normalized` is a fast
  first-pass complexity coordinate for the `chaosscan` classifier.
- Physiologic-style complexity comparisons: `sampleentropy` on
  return series, small `sampleentropy` flags near-periodic regimes.
- Short-window screens: `dispersionentropy` is the preferred entropy
  when the window is a few hundred points, where SampEn is noisy.
- Directed coupling: `transferentropy(x, y)` vs `transferentropy(y, x)`
  gives a lead-lag arrow; pair with `crossrecurrence` for a
  symmetric-plus-directional coupling view.

## Cross-check tests

`tests/sindy/chaos/test_entropy.py`:

- `permutationentropy`: monotone-sequence zero entropy, Gaussian
  normalized entropy above 0.95, rejection on `m = 1` and on
  too-short inputs.
- `sampleentropy`: positive on noise, strictly smaller on a sinusoid
  than on noise at the same `m`, rejection on negative `r`.
- `approximateentropy`: positive on noise, rejection on too-short
  inputs.
- `dispersionentropy`: `normalized` inside `[0, 1]`, Gaussian
  normalized above 0.85, rejection on `c = 1`.
- `transferentropy`: independent series give small TE, coupled series
  give `TE(x -> y) > TE(y -> x)`, rejection on length mismatch and
  under-length inputs.

## Related kernels

- `kuant.sindy.chaos.embedding.mutualinfo`: symmetric information
  between two variables at a lag; TE is the directed analogue.
- `kuant.sindy.chaos.crossrecurrence`: symmetric coupling diagnostic
  in state space rather than in histogram space.
- `kuant.sindy.chaos.chaosscan`: composite regime classifier that
  ingests permutation entropy and Lyapunov together.

## References

- Bandt & Pompe 2002, "Permutation entropy: a natural complexity
  measure for time series," Physical Review Letters 88.
- Richman & Moorman 2000, "Physiological time-series analysis using
  approximate entropy and sample entropy," American Journal of
  Physiology 278.
- Pincus 1991, "Approximate entropy as a measure of system
  complexity," PNAS 88.
- Rostaghi & Azami 2016, "Dispersion entropy: a measure for
  time-series analysis," IEEE Signal Processing Letters 23.
- Schreiber 2000, "Measuring information transfer," Physical Review
  Letters 85.

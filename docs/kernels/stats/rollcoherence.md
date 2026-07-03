# rollcoherence — Rolling Welch coherence in a target band

## Purpose

At each trailing anchor, compute the magnitude-squared coherence
between two series via Welch's method, then return the mean value
inside a specified frequency band.

```math
|C_{xy}(f)|^2 = \frac{|G_{xy}(f)|^2}{G_{xx}(f) \, G_{yy}(f)}
```

## Public API

```python
from kuant.stats import rollcoherence

c = rollcoherence(x, y, window, nperseg=None, band=(0.0, 0.5), fs=1.0)
```

- `x`, `y` — 1D arrays, equal length.
- `window` — trailing window length.
- `nperseg` — Welch segment length; defaults to `window // 2`.
- `band` — `(lo, hi)` in cycles/sample.
- `fs` — sampling frequency, default 1.0 (interpret band in cycles/period).

## Uses

- Regime shifts: coherence with a proxy dropping signals decoupling
- Signal validation: is X actually driving Y at your frequency of interest?
- Portfolio hedge quality: coherence with a hedge instrument

## Design decisions

### Wraps scipy.signal.coherence

We call scipy for the Welch spectrum. scipy is a hard dependency of
kuant, so this always works. If Welch returns bad values for a specific
window (rare), that anchor gets NaN.

### Warm-up NaN, strict-window NaN

`window - 1` bars are NaN. Windows containing NaN in either series get NaN.

## Related

- kuant.stats has no other cross-series primitive at this scale;
  `rollcorr` is the linear-in-time analog (single band collapsed).

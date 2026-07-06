# lyapunov: Rosenstein-Collins-DeLuca largest Lyapunov exponent

## Purpose

Estimate the largest Lyapunov exponent `lambda_1` of a time-delay
embedded trajectory:

    |delta(t)| ~ |delta(0)| * exp(lambda_1 * t)

Sign is the diagnostic:

- `lambda_1 > 0`: exponential divergence of nearby orbits, the
  fingerprint of chaos.
- `lambda_1 ~ 0`: periodic or quasi-periodic dynamics.
- `lambda_1 < 0`: a stable attractor (converges).

Rosenstein 1993 fits the initial linear rise of the mean log-distance
between nearby-neighbor pairs as they evolve forward in time. It is
the practical method for small-to-medium samples (a few hundred to a
few tens of thousands of observations), which covers the range of
most financial series.

## Public API

```python
from kuant.sindy.chaos import lyapunov

lyap = lyapunov(x, tau=1, m=5)
print(lyap.summary())
lyap.lyapunov            # nats per sample
lyap.log_divergence      # full curve, for visual inspection
```

Signature:

```python
lyapunov(
    x, *, tau=1, m=5,
    max_t=None, theiler_window=None,
    fit_start=1, fit_end=None,
)
```

Returns `LyapunovResult` with `lyapunov`, `intercept`, `slope_range`,
`log_divergence`, `embed_dim`, `embed_tau`.

## Design decisions

### `theiler_window = m * tau` default

The Theiler window guards against picking a temporally-close point
as a "neighbor" of the seed. The default `m * tau` is exactly one
embedding window: the minimum separation at which two embedded rows
are guaranteed to be composed of disjoint underlying samples.

### `fit_start=1` skips the initial-condition point

The Rosenstein curve begins at `t=0`, where every seed is paired
with its own nearest neighbor by construction. That first point
sits below the divergence line and would flatten the fitted slope
if included. Fitting from `t=1` reads the slope from actual
divergence, not from the definition of "neighbor."

### `fit_end` defaults to `max_t // 2`

The linear region ends when neighbor pairs saturate at the attractor
diameter. Half the tracked horizon is a defensible default. If the
returned curve visibly bends before that midpoint, refit with a
smaller `fit_end`.

### `max_t = min(N // 4, 40)` default

Enough steps to see the linear rise, few enough that most seed
pairs are still valid at `t = max_t`. Capping at 40 keeps the
inner loop cheap on longer series.

### 200-observation floor

Below 200 finite values, the fit region contains too few valid
pair-steps to give a stable slope.

## When it fires

- The primary "is this chaotic?" test.
- Called by `chaosscan` for the regime classifier's Lyapunov input
  (threshold `> 0.001` for the chaotic verdict).
- The log-divergence curve is unreadable (no linear region, or a
  curve that rises and then falls before saturating) when the
  underlying series is too noisy or too short. Plot
  `log_divergence` before trusting the scalar `lyapunov` value.

## References

- Rosenstein, Collins & DeLuca 1993, "A practical method for
  calculating largest Lyapunov exponents from small data sets."

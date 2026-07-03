# kuant kernel documentation

One doc per kernel, organized by category to mirror the `kuant.*`
subpackage layout.

## Layout

```
docs/kernels/
├── core/     Black-Scholes family + normal-distribution primitives
├── options/  Option analytics on top of core (impvol solver, ...)
├── stats/    Rolling and windowed statistical primitives
├── qm/       QM-inspired tools (HMM, Bell inequality, Zeno scan)
└── sindy/    SINDy-adjacent null-testing tools
```

See [`docs/design/`](../design/) for cross-cutting architecture
decisions, and [`docs/examples/`](../examples/) for worked examples.

## Shared conventions

Every kernel in kuant follows the same contract unless a doc says
otherwise:

- **Backend preserved** — numpy in → numpy out; cupy in → cupy out
- **Dtype preserved** — float32 in → float32 out; ints promote to float64
- **Shape preserved** — broadcasting; scalar-in/scalar-out
- **NaN propagates cleanly** — strict-window semantics for rolling ops;
  passthrough for pointwise ops
- **CPU and GPU parity** — verified in tests

## kuant.core — foundation primitives + Black-Scholes family

### Foundation

| Kernel | Formula | Doc |
|---|---|---|
| [`normcdf`](core/normcdf.md) | `Φ(x) = P[Z ≤ x]` | 28 tests |
| [`normpdf`](core/normpdf.md) | `φ(x) = exp(-x²/2) / √(2π)` | 14 tests |

### Prices

| Kernel | Formula | Doc |
|---|---|---|
| [`bsput`](core/bsput.md) | `K·e^(-r·T)·Φ(-d2) - S·e^(-q·T)·Φ(-d1)` | 23 tests |
| [`bscall`](core/bscall.md) | `S·e^(-q·T)·Φ(d1) - K·e^(-r·T)·Φ(d2)` | 24 tests |

Related by put-call parity: `C - P = S·e^(-q·T) - K·e^(-r·T)`.

### First-order Greeks (direction-specific)

| Kernel | Formula | Range | Doc |
|---|---|---|---|
| [`bsputdelta`](core/bsputdelta.md) | `-e^(-q·T)·Φ(-d1)` | `[-1, 0]` | 24 tests |
| [`bscalldelta`](core/bscalldelta.md) | `+e^(-q·T)·Φ(d1)` | `[0, 1]` | 21 tests |
| [`bsputrho`](core/bsputrho.md) | `-T·K·e^(-r·T)·Φ(-d2)` | `(-∞, 0]` | 18 tests |
| [`bscallrho`](core/bscallrho.md) | `+T·K·e^(-r·T)·Φ(d2)` | `[0, +∞)` | 19 tests |

### Second-order Greeks (put-call symmetric)

| Kernel | Formula | Range | Doc |
|---|---|---|---|
| [`bsgamma`](core/bsgamma.md) | `e^(-q·T)·φ(d1) / (S·σ·√T)` | `[0, +∞)` | 16 tests |
| [`bsvega`](core/bsvega.md) | `S·e^(-q·T)·φ(d1)·√T` | `[0, +∞)` | 15 tests |

## kuant.options — option analytics

| Kernel | Purpose | Doc |
|---|---|---|
| [`impvol`](options/impvol.md) | Vectorized Newton-Raphson implied-vol solver | ~30 tests |

## kuant.stats — rolling and windowed statistics

18 kernels, 245 tests. All strict-window NaN policy. Full list:

| Kernel | Doc |
|---|---|
| [`rollmean`](stats/rollmean.md) · [`rollsum`](stats/rollsum.md) | additive |
| [`rollstd`](stats/rollstd.md) · [`rollmoments`](stats/rollmoments.md) (skew + kurt) | moments |
| [`rollcorr`](stats/rollcorr.md) · [`rollcov`](stats/rollcov.md) · [`rollbeta`](stats/rollbeta.md) · [`rollidio`](stats/rollidio.md) | pairwise |
| [`rollminmax`](stats/rollminmax.md) (min + max) · [`rollrange`](stats/rollrange.md) · [`rollargminmax`](stats/rollargminmax.md) (argmin + argmax) | extremes |
| [`rollquantile`](stats/rollquantile.md) (+ median + percentile) · [`rollrank`](stats/rollrank.md) · [`rollmad`](stats/rollmad.md) | order stats |
| [`rollema`](stats/rollema.md) · [`rollemastd`](stats/rollemastd.md) | recurrence |
| [`zscore`](stats/zscore.md) | composed |

## kuant.qm — QM-inspired tools

| Kernel | Purpose | Doc |
|---|---|---|
| [`hmm`](qm/hmm.md) | Forward / backward / Viterbi / posterior (log-space) | 13 tests |
| [`belltest`](qm/belltest.md) | Bell-inequality-style aggregation test | — |
| [`zenoscan`](qm/zenoscan.md) | Retrain-frequency scan (Zeno-effect) | 3 tests |

## kuant.sindy — SINDy-adjacent null-testing tools

| Kernel | Purpose | Doc |
|---|---|---|
| [`permtest`](sindy/permtest.md) | Permutation-based null-hypothesis test | 5 tests |
| [`grangerscan`](sindy/grangerscan.md) | Bonferroni-corrected Granger F-test scan | — |

## Cross-check testing pattern

Every kernel is validated three ways:

1. **Golden values** — hand-picked reference points, pasted into
   `pytest.parametrize`. Catches typos in the formula.
2. **Reference match against a battle-tested library** — pandas for
   stats, scipy for math, statsmodels for stats tests. Random samples
   matched to `atol=1e-10` typical.
3. **Cross-kernel identities** — put-call parity for BS, finite-
   difference cross-checks between Greeks, shift/scale invariance
   for the rolling family, symmetry `argmax(-x) == argmin(x)`, etc.

**Total across kuant: 516 tests.**

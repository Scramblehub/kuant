# kuant examples

End-to-end usage patterns showing how to compose kernels for real tasks.

## Conventions

Each example lives in its own file and:

1. Imports only from kuant (no other quant deps beyond numpy)
2. Solves one specific task
3. Has a top-of-file docstring explaining what it demonstrates
4. Runs on CPU by default; GPU acceleration is optional
5. Fully self-contained — no external data needed

Run any example directly:

```bash
python docs/examples/<name>.py
```

## Examples

### Options (`kuant.core` + `kuant.options`)

- **[`bs_price_surface.py`](bs_price_surface.py)** — vectorized Black-Scholes
  pricing on a full (strike, tenor) grid, batched Greeks across the same grid,
  put-call and delta parity checks. Shows how a 168-option surface is one
  function call.

- **[`iv_surface_from_market.py`](iv_surface_from_market.py)** — invert a
  synthetic option chain to recover the implied vol surface using
  `impvol` (Newton), with `impvolbisection` as the flat-vega fallback.
  Demonstrates when to use which solver.

### Rolling statistics (`kuant.stats`)

- **[`rolling_zscore_signal.py`](rolling_zscore_signal.py)** — turn a
  return series into a bounded mean-reversion signal via `zscore`.
  Shows equivalence to `(returns - rollmean) / rollstd` and how the
  signal composes with a bounded position-sizing kernel.

### Regime discovery (`kuant.qm`)

- **[`hmm_regime_decode.py`](hmm_regime_decode.py)** — Gaussian-emission
  HMM: simulate a hidden regime process, generate returns, then use
  `viterbi` for the maximum-likelihood state sequence and `posterior`
  for smoothed per-bar state probabilities. Achieves ~96% state
  recovery on the toy data.

### Fat-tail modeling (`kuant.core` fat-tail primitives)

- **[`evt_tail_fit.py`](evt_tail_fit.py)** — Peaks-Over-Threshold fit
  using the Generalized Pareto Distribution (`gpdpdf` / `gpdcdf` /
  `gpdppf`), applied to fat-tailed synthetic returns. Extrapolates
  return-period losses beyond the sample. Compares with Student-t
  (`tcdf`) as a parametric alternative.

### Signal validation (`kuant.sindy`)

- **[`permtest_your_signal.py`](permtest_your_signal.py)** — universal
  permutation null-test via `permtest`. Runs the same test on a real
  linear signal and a pure-noise null; shows p ~ 0.001 for the real
  signal, p ~ 0.82 for the null.

## Next examples

Ideas for future additions:

- `belltest_your_features.py` — classical-bound test on your own factor pairs
- `granger_signal_scan.py` — screen a macro library against your target
- `sindylasso_feature_discovery.py` — LASSO scan across a hand-engineered library
- `impvol_vs_bisection.py` — timing benchmark, when each pays off

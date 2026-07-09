# kuant roadmap — future additions

Living list of things planned but not yet built. Priority-ordered
within each category. If you're reading this and want to build one,
open an issue first so we can align on scope.

## Scheduled next

- **kuant.topology** — see [plan-topology.md](plan-topology.md)
- **kuant.options + kuant.core Greeks** — see [plan-options.md](plan-options.md)

## kuant.qm

- **hmm.baumwelch** — EM parameter training for discrete HMM. Given
  `obs`, `n_states`, `n_symbols`, `max_iter`, iterate E-step
  (γ, ξ via `posterior`) and M-step (update π, A, B). Returns
  learned parameters + log-likelihood history.
- **ghmm.baumwelch** — Same for the Gaussian HMM. M-step updates
  are `μ_i = Σ_t γ[t,i]·obs[t] / Σ_t γ[t,i]` etc.
- **hmmensemble** — Temporal Ensemble HMM:
  research). Fits K HMMs with different initializations and averages
  their posteriors. Sibling of `nocloningscan`.
- **densitymatrix** — Full N×N uncertainty representation instead of
  the diagonal `γ` vector. Captures regime ambiguity better.
- **wignerscan** — 2D Wigner-function-style distribution over
  (feature, momentum) for two-feature systems. Look for negative
  regions (non-classicality).
- **entanglementscan** — Pair-trading residual correlation dynamics
  over time as an entanglement analog.
- **tunnelscan** — Discrete-jump detection in factor exposures (vs
  smooth transitions). Test whether "regime" boundaries are
  probability barriers or hard cliffs.

## kuant.sindy

- **sindylasso** — LASSO-with-CV feature library scan. Given a target
  and a feature library, run cross-validated LASSO and return the
  selected non-zero coefficients + optional built-in permutation
  test. Baseline linear feature-library scan.
- **pinnscan** — Nonlinear counterpart to sindylasso. GradientBoosting
  on the library + permutation p-value on the OOF predictions.
  Nonlinear counterpart to sindylasso.
- **symbolicscan** — Polynomial-symbolic regression scan. Bivariate
  degree-2 by default. Return the sparsest polynomial that beats a
  baseline R².
- **crossderivativescan** — 2D bivariate ramp fits (`∂²y / ∂x₁∂x₂`).

- **tvsindy** — Time-varying SINDy fit with rolling windows. Distilled
  .
- **accelerationscan** — `d²y/dt²` predictive power scan. Distilled
  .

## kuant.topology

- **persistenthomology** — Persistent homology on a time-series or
  point cloud. Return birth/death pairs.
- **wasserstein** — Wasserstein distance between persistence diagrams.
- **bettinumbers** — Betti number time-series (b₀, b₁, ...).
- **mapper** — Mapper algorithm on high-dim data with a scalar filter
  function.
- **dispersioncollapse** — Sector dispersion collapse signal (per prior
  bubble diagnostic S3, weak but worth including as reference).

## kuant.core

- **bscalltheta**, **bsputtheta** — Time-decay Greeks (`∂price/∂t`).
- **bscallcharm**, **bsputcharm** — `∂delta/∂t` (delta decay).
- **impvolvega** — Newton-Raphson on the vega-weighted objective for
  improved convergence in the low-vega tail.

## kuant.stats

- **rollema** with `adjust=True` mode.
- **rollrank** with `method='dense'` and `method='ordinal'` variants
  (currently only average-rank).
- **rollacf** — rolling autocorrelation at lag k.
- **rollentropy** — Shannon entropy of a rolling histogram.

## kuant.options

- **impvolbisection** — Bisection fallback for the low-vega tail
  where Newton diverges.
- **optionchain** — Utilities for building/filtering option chains
  (strike-tenor grids, moneyness bands, expiry filters).

## kuant.portfolio

Empty subpackage. First candidates:

- **sharperatio** — Rolling and full-history Sharpe.
- **drawdown** — Peak-to-trough drawdown series + max drawdown.
- **calmarratio** — CAGR / |MaxDD|.
- **turnover** — Rolling position turnover.
- **contribution** — Per-asset P&L attribution.

## kuant.backtest

Empty subpackage. First candidates:

- **walkforward** — Walk-forward CV harness (matches the pattern in
  zenoscan and decoherencescan).
- **triplebarrier** — Meta-labeling triple-barrier method (Lopez de
  Prado).
- **transactioncost** — Slippage and commission modeling.

## kuant.signals

Empty subpackage. First candidates:

- **breadthscore** — Cross-sectional breadth signal generator.
- **momentumfamily** — Assorted momentum flavors (Jegadeesh, Carhart,
  Asness, ...).

## kuant.text

Empty subpackage. First candidates:

- **occparse** — Parse OCC option symbols to (underlying, expiry, C/P,
  strike).
- **secformparse** — Parse SEC form types and filing headers.
- **lmdict** — Loughran-McDonald finance lexicon lookup.

## kuant.data

Empty subpackage. First candidates:

- **baragg** — Bar aggregation (1m → 5m, 1d → 1w, ...).
- **align** — Multi-series alignment on a common calendar.
- **corpaction** — Corporate action adjustment (splits, dividends).

## kuant.edgecases

Empty subpackage. First candidates:

- **nanpolicies** — Explicit NaN-handling policies as callable
  strategies (strict, skipna, forwardfill, ...).
- **delistedhandling** — Utilities for delisted-name handling in
  historical backtests.

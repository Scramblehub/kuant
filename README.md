# kuant

[![PyPI version](https://img.shields.io/pypi/v/kuant.svg)](https://pypi.org/project/kuant/)
[![Python versions](https://img.shields.io/pypi/pyversions/kuant.svg)](https://pypi.org/project/kuant/)
[![CI](https://github.com/Scramblehub/kuant/actions/workflows/ci.yml/badge.svg)](https://github.com/Scramblehub/kuant/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/Scramblehub/kuant/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/pypi/l/kuant.svg)](https://github.com/Scramblehub/kuant/blob/main/LICENSE)

A Python library of numeric kernels for quantitative research.
Works on numpy, dispatches transparently to cupy when available.

## What's in the box

Fifteen subpackages, 1656 tests, ~90 benchmarks. Alpha stability.

| Subpackage | Kernels | Contents |
|---|---|---|
| `kuant.core` | 16 | Black-Scholes pricing, Gaussian family (cdf/pdf/ppf plus log-tails), Student-t family, Generalized Pareto family, logsumexp. |
| `kuant.options` | 21 | First- and second-order Greeks, payoffs, chain filters, implied-vol via Newton or bisection. |
| `kuant.stats` | 29 | Rolling primitives with strict-window NaN semantics, Hurst R/S, rolling Hurst, risk metrics (Sharpe, Sortino, MDD, Calmar), realized volatility, stationarity tests, tail cluster (Hill tail index, rolling variant, DFA, rollcoherence). |
| `kuant.qm` | 5 plus `hmm`/`ghmm` submodules | HMM and Gaussian-HMM inference (forward, backward, viterbi, posterior) plus Baum-Welch EM training. Also belltest, zenoscan, posteriorentropy, nocloningscan, decoherencescan. |
| `kuant.sindy` | 6 | permtest, grangerscan, sindylasso, pinnscan, symbolicscan, accelerationscan. |
| `kuant.topology` | 4 | persistenthomology (ripser), bettiseries, wasserstein (persim), dispersioncollapse. |
| `kuant.data` | 6 | align (inner/outer/forward), baragg (OHLCV), corpaction (splits/dividends), panelize, resample, stitch. |
| `kuant.edgecases` | 3 | Five NaN policies, delisted-name utilities, an outlier detector with three methods. |
| `kuant.signals` | 4 | winsorize, neutralize (OLS residual), icdecay (Spearman IC decay curve), factorscoring. |
| `kuant.portfolio` | 5 | P&L contribution, drawdown, Sharpe, Sortino, aggregate risk metrics. |
| `kuant.text` | 4 | tickernorm, cusipvalidate, occparse (option symbols), secformparse. |
| `kuant.nulltest` | 3 | bootstrap, multiple-hypothesis correction, White/Hansen SPA test. |
| `kuant.queueing` | 2 | hardware throttle, request-coordination layer. |
| `kuant.backtest` | 2 subpackages | `lifecycle` (SecurityLifecycle + TerminalAction + apply_lifecycle + tradeable_mask + lifecycle_returns + detect_delistings); `liquidity` (LiquidityProfile + FlatSlippage / LinearImpact / SquareRootImpact + execute_fill + liquidity_mask). `fill`, `position`, `warmup`, `engine` planned. |

Each kernel has: an API doc in [`docs/kernels/`](docs/kernels/), a numpy
implementation, a cupy path where the math is batched, and a test suite
against golden values, library references, and cross-kernel identities.
Errors go through the `kuant.errors` hierarchy; every message names the
kernel, the offending value, a stable error code, and a one-line fix.

## Positioning

**Lifecycle correctness.** `kuant.backtest.lifecycle` ships first-class
listing and delisting semantics: `SecurityLifecycle` and
`TerminalAction` (`LIQUIDATE_AT_LAST`, `MARK_TO_ZERO`,
`PRORATE_RECOVERY`), plus `apply_lifecycle`, `tradeable_mask`,
`lifecycle_returns`, and `detect_delistings`. Most backtest engines
silently ignore orders on NaN prices, or forward-fill the last live
price forever. Both quietly corrupt returns on real point-in-time
equity books. kuant treats the tradeable window as a first-class
primitive that simulators consult BEFORE trying to fill, which closes
a silent-corruption gap that most backtest engines share.

**Numerical correctness.** kuant's rolling stats use shifted-cumsum
arithmetic instead of the naive `E[X²] - E[X]²` identity with
absolute-value masking. Near-constant series with large additive
offsets recover epsilon-scale standard deviations instead of blowing
up into NaNs or spurious multi-percent readings. The guarantee is
enforced by an adversarial test suite in
`tests/stats/test_numerical_stability.py` that covers near-constant
series, large additive offsets, long slow drifts, alternating signs,
higher moments, and pandas-parity guardrails.

## What's new in v0.3.x

- **v0.3.0** (yanked): `kuant.text.tickernorm`, `kuant.signals.factorscoring`,
  tearsheet parity across `kuant.portfolio`, `kuant.stats.realizedvol`,
  `kuant.stats.stationarity`, and the initial `kuant.nulltest` cluster
  (bootstrap, MHT correction, SPA test).
- **v0.3.1**: lifecycle primitives land with `SecurityLifecycle`,
  `TerminalAction`, `apply_lifecycle`, `tradeable_mask`,
  `lifecycle_returns`, `detect_delistings`, and a paired identifier
  scrub in `kuant.text` for tickernorm plus CUSIP validation.
- **v0.3.2**: `rollemastd` picks up the shifted-cumsum fix that
  eliminates catastrophic cancellation on near-constant inputs, and
  ships alongside the adversarial numerical-stability test suite
  described in Positioning.
- **v0.4.0**: lifecycle moves from `kuant.lifecycle` to
  `kuant.backtest.lifecycle` under the new `kuant.backtest` umbrella
  for correctness-first backtest primitives. `kuant.lifecycle` remains
  as a deprecation shim through 0.4.x and is removed in 0.5.0.
- **v0.4.1**: `kuant.backtest.liquidity` lands with `LiquidityProfile`
  (ADV, spread, min_size, max_participation), three fill models
  (`FlatSlippage`, `LinearImpact`, `SquareRootImpact` for Almgren-Chriss),
  `execute_fill` + `execute_fill_panel` with categorical `FillResult`
  reasons (OK, CAPPED_PARTICIPATION, BELOW_MIN_SIZE, NO_LIQUIDITY,
  MISSING_DATE), and `liquidity_mask` for composing with lifecycle's
  `tradeable_mask`.

## Install

```bash
pip install kuant                 # CPU only
pip install kuant[gpu]            # adds cupy-cuda12x
pip install kuant[topology]       # adds ripser, persim
pip install kuant[sindy]          # adds scikit-learn, statsmodels
pip install kuant[all]            # all optional bundles
```

## Shared kernel contract

Every kernel in the library obeys the same six-part contract:

1. **CPU-first, GPU-second.** Every kernel has a numpy fallback.
   Development works on any machine; cupy accelerates when present.
2. **Batched by default.** Kernels take arrays, not scalars. Scalars
   are a special case, not the base case.
3. **Explicit edge cases.** NaN, zero denominators, empty arrays, and
   past-expiry options are handled inside the kernel. Callers do not
   need defensive wrappers.
4. **Composable primitives.** Each kernel does one thing. Complex
   operations build from atoms.
5. **Reproducible.** Kernels do not touch global state. Same inputs,
   same outputs.
6. **No underscores in the API surface.** `bsput`, not `bs_put`.
   `belltest`, not `bell_test`.

Errors follow the same shape everywhere. Every failure raises a
`KuantError` subclass with the offending value, a stable error code,
and a one-line fix. Runtime warnings follow the same shape via
`KuantWarning`. See
[`docs/design/Validation_Additions.md`](docs/design/Validation_Additions.md).

Tabular results ship a `.to_parquet(path)` method via lazy `pyarrow`.
No CSV or JSON helpers.

## Benchmarks

Median wall time on batches of 1M elements, measured on one NVIDIA GPU
and one Intel-class CPU:

| Kernel | numpy | cupy | speedup |
| --- | ---: | ---: | ---: |
| `bscall`   1M | 92.5 ms | 1.13 ms | 82x |
| `normcdf`  1M | 14.4 ms | 0.17 ms | 85x |

Below ~10K elements the dispatch overhead dominates; above that the GPU
runs 50-100x faster. The full suite (~90 measurements) lives in
`benchmarks/`. To reproduce:

```bash
./benchmarks/run.sh
tail -f benchmarks/results/latest.jsonl
```

The runner reads new files in `benchmarks/suites/` at the next scan, so
you can queue additions while a run is in progress. See
[`benchmarks/README.md`](benchmarks/README.md) for the workflow.

## Documentation

- [`docs/kernels/`](docs/kernels/): one API doc per kernel, grouped by subpackage.
- [`docs/design/`](docs/design/): cross-cutting design decisions.
- [`docs/examples/`](docs/examples/): worked examples.

Start at [`docs/README.md`](docs/README.md).

## Quick start

```python
import numpy as np
import pandas as pd
from kuant.core import bsput
from kuant.stats import zscore
from kuant.options import impvol
from kuant.qm.hmm import viterbi
from kuant.backtest.lifecycle import (
    SecurityLifecycle,
    TerminalAction,
    tradeable_mask,
    lifecycle_returns,
)

# Vectorized Black-Scholes put pricing
strikes = np.linspace(80, 120, 41)
prices = bsput(S=100.0, K=strikes, T=1.0, r=0.05, sigma=0.20)

# Invert market prices to implied vol
sigma_iv = impvol(prices, S=100.0, K=strikes, T=1.0, r=0.05, is_call=False)

# Rolling z-score of returns
z = zscore(returns, window=252)

# HMM state decoding
states, log_prob = viterbi(observations, pi, A, B)

# First-class listing and delisting semantics
lc = SecurityLifecycle(
    symbol="XYZ.12345",
    listing_date=pd.Timestamp("2010-01-04"),
    delisting_date=pd.Timestamp("2019-06-14"),
    terminal_action=TerminalAction.PRORATE_RECOVERY,
    recovery_fraction=0.42,
)
gate = tradeable_mask(price_series, lc)   # simulators consult THIS
r = lifecycle_returns(price_series, lc)   # terminal transition baked in
```

## Repository layout

```
kuant/
├── core/         BS family, normal CDF/PDF, log-space primitives
├── options/      Greeks, impvol solvers, chain filters
├── stats/        Rolling and windowed statistics
├── qm/           HMM/GHMM plus regime-discovery tools
├── sindy/        SINDy-adjacent null-testing
├── topology/     Persistent homology, betti series, wasserstein, dispersioncollapse
├── data/         align, baragg, corpaction, panelize, resample, stitch
├── edgecases/    NaN policies, delisted handling, outlier detection
├── signals/      winsorize, neutralize, icdecay, factorscoring
├── portfolio/    P&L contribution, drawdown, Sharpe, Sortino, risk metrics
├── text/         tickernorm, cusipvalidate, occparse, secformparse
├── nulltest/     bootstrap, MHT correction, White/Hansen SPA test
├── lifecycle/    SecurityLifecycle, tradeable_mask, lifecycle_returns
├── backtest/     Simulation engine (scaffold; v1 next)
├── queueing/     Hardware throttle and coordination layer
├── errors.py     KuantError + KuantWarning hierarchies
└── _validation.py  Central validators used by every kernel

docs/
├── kernels/      One doc per kernel
├── design/       Cross-cutting design decisions
└── examples/     Worked examples

tests/            1:1 with kernel files; 1656 tests total
```

## Contributing

Guidelines will land as the project stabilizes.

## License

Apache 2.0. See [LICENSE](LICENSE).

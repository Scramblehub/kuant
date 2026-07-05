# kuant

[![PyPI version](https://img.shields.io/pypi/v/kuant.svg)](https://pypi.org/project/kuant/)
[![Python versions](https://img.shields.io/pypi/pyversions/kuant.svg)](https://pypi.org/project/kuant/)
[![CI](https://github.com/Scramblehub/kuant/actions/workflows/ci.yml/badge.svg)](https://github.com/Scramblehub/kuant/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/Scramblehub/kuant/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/pypi/l/kuant.svg)](https://github.com/Scramblehub/kuant/blob/main/LICENSE)

GPU-accelerated quantitative research kernels. *kernel × quant.*

## Vision

Standard quant libraries stop at technical indicators and portfolio math.
kuant additionally exposes:

- **Sparse Identification of Nonlinear Dynamics (SINDy)** — null-testing
  tools for feature discovery, distilled from real production research
- **Quantum-inspired regime discovery** — HMM state machines, Bell-
  inequality-style aggregation tests, retrain-frequency (Zeno) scans
- **Topological Data Analysis (TDA)** — persistent homology, Mapper
  algorithm *(planned)*

Combined with GPU-batched primitives (Black-Scholes, rolling statistics,
implied-vol solvers), kuant is a research-grade toolkit for signal
discovery — not just an implementation of textbook indicators.

## Status

Alpha. **1325 tests** across 9 shipped subpackages:

| Subpackage | Kernels | Highlights |
| --- | --- | --- |
| `kuant.core` | 16 | BS pricing (bscall/bsput) + full Gaussian family (normcdf/normpdf/normppf + log-tails) + Student-t (tcdf/tpdf/tppf + log-tails) + Generalized Pareto (gpdcdf/gpdpdf/gpdppf) + logsumexp |
| `kuant.options` | 21 | First-order Greeks (delta/gamma/vega/rho/theta/charm) + second-order (vanna/volga/speed/zomma/color) + payoffs + chain filters + Newton and bisection implied-vol solvers |
| `kuant.stats` | 27 | Rolling primitives with strict-window NaN + Hurst (R/S) + rolling Hurst + risk metrics (Sharpe/Sortino/MDD/Calmar) + tail cluster (Hill tailindex, rolling variant, DFA, rollcoherence) |
| `kuant.qm` | 5 + `hmm`/`ghmm` subpackages | HMM/GHMM inference (forward/backward/viterbi/posterior) + **Baum-Welch EM training** + belltest, zenoscan, posteriorentropy, nocloningscan, decoherencescan |
| `kuant.sindy` | 6 | permtest, grangerscan, sindylasso, pinnscan, symbolicscan, accelerationscan |
| `kuant.topology` | 4 | persistenthomology (ripser) + bettiseries + wasserstein (persim) + dispersioncollapse |
| `kuant.data` | 6 | align (inner/outer/forward) + baragg (OHLCV) + corpaction (splits/dividends) + panelize + resample + stitch |
| `kuant.edgecases` | 6 | nanpolicies (5 strategies) + delistedhandling (zero/hold/recovery) + outlierpolicy (mad/iqr/zscore) |
| `kuant.signals` | 3 | winsorize + neutralize (OLS residual) + icdecay (Spearman IC decay curve) |

Each kernel has:

- Full API doc under [`docs/kernels/`](docs/kernels/)
- CPU fallback (numpy path — works on any machine)
- GPU path (cupy — same math, verified for parity) where the math batches
- Cross-checked test suite (golden values, library reference, cross-kernel identities, machine-precision fd tolerances)
- Informative errors and runtime warnings via `kuant.errors`
  ([`KuantValueError`](kuant/errors.py), `KuantNumericWarning`, …) — every message names the kernel, the actual bad value, a stable error code, and a concrete fix line

## Install

```bash
# CPU-only
pip install kuant

# With GPU
pip install kuant[gpu]

# Bundle for topological data analysis (ripser + persim)
pip install kuant[topology]

# Bundle for SINDy tools (scikit-learn + statsmodels)
pip install kuant[sindy]

# All optional bundles at once
pip install kuant[all]
```

## Benchmarks

GPU acceleration on batched primitives is real. Verified median wall time
on batches of 1M elements (NVIDIA GPU + Intel-class CPU):

| Kernel              | numpy    | cupy    | speedup |
| ---                 | ---:     | ---:    | ---:    |
| `bscall`   1M       | 92.5 ms  | 1.13 ms | **82x** |
| `normcdf`  1M       | 14.4 ms  | 0.17 ms | **85x** |

Smaller batches are dispatch-overhead bound; the crossover is around 10K
elements. The whole benchmark suite (~90 measurements across the 9
subpackages) is in `benchmarks/`. Reproduce with:

```bash
./benchmarks/run.sh
tail -f benchmarks/results/latest.jsonl
```

The runner is queue-based — you can drop a new benchmark file into
`benchmarks/suites/` while the queue is running, and it gets picked up
on the next scan. See `benchmarks/README.md` for the full workflow.

## Documentation

- [`docs/kernels/`](docs/kernels/) — per-kernel API docs, one per kernel,
  organized into `core/`, `options/`, `stats/`, `qm/`, `sindy/`
- [`docs/design/`](docs/design/) — cross-cutting design decisions
- [`docs/examples/`](docs/examples/) — worked examples

Start at [`docs/README.md`](docs/README.md).

## Quick start

```python
import numpy as np
from kuant.core import bsput, bsputdelta, normcdf
from kuant.stats import rollmean, rollstd, zscore, rollcorr
from kuant.options import impvol
from kuant.qm.hmm import forward, viterbi

# Vectorized Black-Scholes put pricing
strikes = np.linspace(80, 120, 41)
prices = bsput(S=100.0, K=strikes, T=1.0, r=0.05, sigma=0.20)

# Invert market prices to implied vol
sigma_iv = impvol(prices, S=100.0, K=strikes, T=1.0, r=0.05, is_call=False)

# Rolling z-score of returns
z = zscore(returns, window=252)

# HMM state decoding
states, log_prob = viterbi(observations, pi, A, B)
```

## Repository layout

```folder
kuant/
├── core/         Mathematical primitives (BS family, normal CDF/PDF)
├── options/      Options analytics (Greeks, impvol solvers, chain filters)
├── stats/        Rolling and windowed statistics (27 kernels)
├── qm/           HMM/GHMM + Baum-Welch + regime-discovery tools
├── sindy/        SINDy-adjacent null-testing (permtest, grangerscan, ...)
├── topology/     TDA (persistent homology, betti, wasserstein, dispersioncollapse)
├── data/         Data-shape primitives (align, baragg, corpaction, panelize, resample, stitch)
├── edgecases/    NaN policies, delisted handling, outlier detection
├── signals/      winsorize, neutralize, icdecay
├── portfolio/    P&L, drawdown, Sharpe (skeleton — v1 next)
├── backtest/     Simulation engine (skeleton)
├── text/         Text parsing (skeleton — v1 next)
├── errors.py     KuantError hierarchy + KuantWarning classes
├── _validation.py Central validators (used by every kernel)
└── queueing/     Hardware throttle + coordination layer

docs/
├── kernels/      One doc per kernel (grouped by subpackage)
├── design/       Cross-cutting design docs
└── examples/     Worked examples

tests/            1:1 with kernel files; 1325 tests total
```

## Design principles

1. **CPU-first, GPU-second** — every kernel has a numpy fallback so
   development works on any machine
2. **Batched by default** — kernels operate on tensors, not scalars;
   single scalar case is a special call
3. **Explicit edge cases** — NaN, zero denominators, empty arrays,
   past-expiry options all handled in-kernel; callers don't need
   defensive wrappers
4. **Composable primitives** — each kernel does one thing; complex
   operations build up from atoms
5. **Reproducible** — kernels don't touch global state; same inputs
   guarantee same outputs
6. **No underscores in the API surface** — `bsput`, not `bs_put`;
   `belltest`, not `bell_test`. Improves typing flow
7. **Informative errors, not black boxes** — every kernel raises a
   `KuantError` subclass with the offending value, a stable error code,
   and a copy-pasteable fix line. Same shape for runtime warnings via
   `KuantWarning`. See [`docs/design/Validation_Additions.md`](docs/design/Validation_Additions.md).
8. **Parquet-first for tabular output** — result dataclasses ship
   `.to_parquet(path)` via lazy `pyarrow`. No CSV/JSON helpers by design.

## Contributing

*Contribution guidelines coming as project matures.*

## License

Apache 2.0. See [LICENSE](LICENSE).

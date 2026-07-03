# kuant

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

Alpha. **955 tests** across 5 shipped subpackages:

| Subpackage | Kernels | Highlights |
| --- | --- | --- |
| `kuant.core` | 16 | BS pricing (bscall/bsput) + full Gaussian family (normcdf/normpdf/normppf + log-tails) + Student-t (tcdf/tpdf/tppf + log-tails) + Generalized Pareto (gpdcdf/gpdpdf/gpdppf) + logsumexp |
| `kuant.options` | 21 | First-order Greeks (delta/gamma/vega/rho/theta/charm) + second-order (vanna/volga/speed/zomma/color) + payoffs + chain filters + Newton and bisection implied-vol solvers |
| `kuant.stats` | 19 | Rolling primitives with strict-window NaN, plus Hurst (R/S) and rolling Hurst |
| `kuant.qm` | 5 + `hmm`/`ghmm` subpackages | HMM/GHMM inference (forward/backward/viterbi/posterior) + belltest, zenoscan, posteriorentropy, nocloningscan, decoherencescan |
| `kuant.sindy` | 6 | permtest, grangerscan, sindylasso, pinnscan, symbolicscan, accelerationscan |

Each kernel has:

- Full API doc under [`docs/kernels/`](docs/kernels/)
- CPU fallback (numpy path — works on any machine)
- GPU path (cupy — same math, verified for parity)
- Cross-checked test suite (golden values, library reference, cross-kernel identities, machine-precision fd tolerances)

## Install

```bash
# CPU-only
pip install kuant

# With GPU
pip install kuant[gpu]

# Some tools have optional heavy dependencies (scikit-learn for belltest,
# statsmodels for grangerscan). Install those separately when needed:
pip install scikit-learn statsmodels
```

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
├── options/      Options analytics (impvol solver)
├── stats/        Rolling and windowed statistics (18 kernels)
├── qm/           Quantum-inspired tools (HMM, belltest, zenoscan)
├── sindy/        SINDy-adjacent null-testing tools (permtest, grangerscan)
├── portfolio/    P&L, drawdown, Sharpe (skeleton)
├── backtest/     Simulation engine (skeleton)
├── signals/      Signal computation (skeleton)
├── topology/     TDA (skeleton)
├── text/         Text parsing (skeleton)
├── data/         Bar aggregation (skeleton)
├── edgecases/    Edge case utilities (skeleton)
└── queueing/     Hardware throttle + coordination layer

docs/
├── kernels/      One doc per kernel (grouped by subpackage)
├── design/       Cross-cutting design docs
└── examples/     Worked examples

tests/            1:1 with kernel files; 887 tests total
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

## Contributing

*Contribution guidelines coming as project matures.*

## License

Apache 2.0. See [LICENSE](LICENSE).

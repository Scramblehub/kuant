# kuant

GPU-accelerated quantitative research kernels. *kernel × quant.*

## Vision

Standard quant libraries stop at technical indicators and portfolio math. kuant
additionally exposes:

- **Sparse Identification of Nonlinear Dynamics (SINDy)** — discover governing
  equations from data
- **Quantum-inspired regime discovery** — HMM state machines, density matrix
  formalism, Bell inequality tests on paired price series
- **Topological Data Analysis (TDA)** — persistent homology, Mapper algorithm,
  Wasserstein distance between persistence diagrams

Combined with GPU-batched primitives (Black-Scholes, rolling statistics,
cross-sectional operations, backtest engine components), kuant is a
research-grade toolkit for signal discovery — not just an implementation of
textbook indicators.

## Status

Alpha. Planning phase. Kernels landing one at a time, each with:

- Full docstring (purpose, signature, edge cases)
- CPU fallback (numpy path for development without GPU)
- GPU kernel (cupy `RawKernel` or dedicated CUDA C++)
- Test suite (golden data, edge cases, benchmarks)
- Real-world validation (verified against research script inline implementations)

## Install

```bash
# CPU-only (development, no GPU needed)
pip install kuant

# With GPU support (requires CUDA toolkit)
pip install kuant[gpu]

# For contributors
pip install kuant[dev]
```

## Quick start

*Coming as kernels land.*

## Structure

```
kuant/
├── core/         Mathematical primitives (BS pricing, norm CDF, returns)
├── options/      Options-specific (pricing, Greeks, chain filters)
├── stats/        Rolling and cross-sectional statistics
├── portfolio/    P&L, drawdown, Sharpe, attribution
├── backtest/     Simulation engine components
├── signals/      Signal computation (regime, VWAP, correlation break)
├── text/         Regex and text parsing (OCC symbols, SEC forms, LM dict)
├── data/         Bar aggregation, alignment, corporate actions
├── edgecases/    NaN handling, sparse trading, delisted names
├── queueing/     Coordination layer (job queue, freshness, dep graph)
├── sindy/        Sparse Identification of Nonlinear Dynamics
├── qm/           Quantum-mechanics-inspired regime discovery
└── topology/     Topological Data Analysis
```

## Design principles

1. **CPU-first, GPU-second** — every kernel has a numpy fallback so
   development works on any machine
2. **Batched by default** — kernels operate on tensors, not scalars; single
   scalar case is a special call
3. **Explicit edge cases** — NaN, zero denominators, empty arrays, past-expiry
   options all handled in-kernel; callers don't need defensive wrappers
4. **Composable primitives** — each kernel does one thing; complex operations
   build up from atoms
5. **Reproducible** — kernels don't touch global state; same inputs guarantee
   same outputs

## Contributing

*Contribution guidelines coming as project matures.*

## License

Apache 2.0. See [LICENSE](LICENSE).

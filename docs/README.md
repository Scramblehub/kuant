# kuant docs

Three top-level directories, each with a single responsibility:

## [`kernels/`](kernels/) — per-kernel documentation

One markdown file per kernel, grouped by category subdirectory:

- [`kernels/core/`](kernels/core/) — Black-Scholes family + normal
  distribution primitives (10 kernels)
- [`kernels/options/`](kernels/options/) — option analytics (impvol solver)
- [`kernels/stats/`](kernels/stats/) — rolling and windowed statistics
  (18 kernels)
- [`kernels/qm/`](kernels/qm/) — QM-inspired tools (HMM, belltest,
  zenoscan)
- [`kernels/sindy/`](kernels/sindy/) — SINDy-adjacent null-testing tools
  (permtest, grangerscan)

Start at [`kernels/README.md`](kernels/README.md) for the full index.

## [`design/`](design/) — cross-cutting design docs

Architectural decisions that span multiple kernels: GPU vs CPU strategy,
testing policy, backend/dtype invariants, dependency management, naming
conventions.

Currently empty — will populate as decisions accumulate.

## [`examples/`](examples/) — worked examples

End-to-end usage patterns showing how to compose kernels to solve a
concrete task. Currently empty — will populate as examples are written.

## Conventions

**Every kernel doc includes:**

- Purpose (one line)
- Public API (import + call signature)
- Design decisions (WHY it works the way it does)
- Edge cases (behavior at boundaries)
- Cross-check tests (how it's validated)
- Related tools

**Every kernel in kuant follows the same contract** unless a doc says otherwise:
backend-preserving, dtype-preserving, shape-preserving, NaN-propagating,
CPU/GPU parity verified.

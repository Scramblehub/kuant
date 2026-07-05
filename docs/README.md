# kuant docs

Three top-level directories.

## [`kernels/`](kernels/): per-kernel documentation

One markdown file per kernel, grouped by subpackage:

- [`kernels/core/`](kernels/core/): Black-Scholes family, Gaussian primitives,
  Student-t and Generalized Pareto tails, logsumexp.
- [`kernels/options/`](kernels/options/): Greeks, payoffs, chain filters,
  implied-vol solvers.
- [`kernels/stats/`](kernels/stats/): rolling and windowed statistics
  (27 kernels).
- [`kernels/qm/`](kernels/qm/): HMM/GHMM inference, Baum-Welch training,
  belltest, zenoscan, decoherencescan.
- [`kernels/sindy/`](kernels/sindy/): permtest, grangerscan, sindylasso,
  pinnscan, symbolicscan, accelerationscan.
- [`kernels/topology/`](kernels/topology/): persistenthomology, bettiseries,
  wasserstein, dispersioncollapse.
- [`kernels/data/`](kernels/data/): align, baragg, corpaction, panelize,
  resample, stitch.
- [`kernels/edgecases/`](kernels/edgecases/): nanpolicies, delistedhandling,
  outlierpolicy.
- [`kernels/signals/`](kernels/signals/): winsorize, neutralize, icdecay.

Start at [`kernels/README.md`](kernels/README.md) for the full index.

## [`design/`](design/): cross-cutting design docs

Architectural decisions that span multiple kernels. GPU vs CPU dispatch,
backend and dtype invariants, error contracts, naming conventions,
scope drafts for future subpackages.

## [`examples/`](examples/): worked examples

End-to-end usage patterns showing how to compose kernels for a concrete
task.

## Conventions

Every kernel doc covers: purpose (one line), public API, design decisions
(why it works this way), edge cases, cross-check tests, related tools.

Every kernel in kuant follows the same contract unless a doc says
otherwise: backend-preserving, dtype-preserving, shape-preserving,
NaN-propagating, CPU/GPU parity verified.

Errors and warnings follow the `kuant.errors` contract. Every failure
raises a `KuantError` subclass with the offending value, a stable error
code, and a one-line fix. Runtime warnings use the same shape via
`KuantWarning`.

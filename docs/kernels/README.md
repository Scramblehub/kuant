# kuant kernel documentation

One doc per kernel, organized by subpackage.

## Layout

```
docs/kernels/
├── core/       Black-Scholes family, Gaussian and Student-t primitives, GPD, logsumexp
├── options/    Greeks, payoffs, chain filters, implied-vol solvers
├── stats/      Rolling and windowed statistical primitives (27 kernels)
├── qm/         HMM/GHMM inference, Baum-Welch training, regime tools
├── sindy/      Null-testing scans (permtest, grangerscan, sindylasso, ...)
├── topology/   Persistent homology, betti series, wasserstein, dispersioncollapse
├── data/       align, baragg, corpaction, panelize, resample, stitch
├── edgecases/  NaN policies, delisted handling, outlier detection
└── signals/    winsorize, neutralize, icdecay
```

See [`docs/design/`](../design/) for cross-cutting decisions and
[`docs/examples/`](../examples/) for worked examples.

## Shared kernel contract

Every kernel follows the same contract unless a doc says otherwise:

- Backend preserved. numpy in, numpy out. cupy in, cupy out.
- Dtype preserved. float32 in, float32 out. Integers promote to float64.
- Shape preserved. Broadcasting. Scalar in, scalar out.
- NaN propagates cleanly. Strict-window semantics for rolling ops.
- CPU and GPU parity verified in tests where the math batches.

## Error contract

Every kernel raises `KuantError` subclasses on caller mistakes:

- `KuantValueError` for out-of-range inputs
- `KuantShapeError` for shape or broadcasting problems
- `KuantConvergenceError` for iterative solvers that fail
- `KuantDependencyError` for missing optional imports

Runtime warnings that flag unreliable-but-computed results use
`KuantWarning` and its subclasses (`KuantConvergenceWarning`,
`KuantNumericWarning`, `KuantOverflowWarning`).

Every message names the kernel, the offending value, a stable code
like `KE-VAL-POSITIVE` or `KW-CV-ENDPOINT-LOW`, and a one-line fix.
See [`../design/Validation_Additions.md`](../design/Validation_Additions.md).

## Cross-check testing pattern

Every kernel is validated three ways:

1. Golden values. Hand-picked reference points fed through
   `pytest.parametrize`. Catches typos in the formula.
2. Reference match against a battle-tested library where one exists.
   pandas for stats, scipy for math, statsmodels for hypothesis tests.
   Random samples matched to `atol=1e-10` typical.
3. Cross-kernel identities. Put-call parity for BS, finite-difference
   cross-checks between Greeks, shift and scale invariance for the
   rolling family, `argmax(-x) == argmin(x)`, etc.

# kuant design docs

Cross-cutting architectural decisions and design rationale that spans
multiple kernels or the library as a whole.

For per-kernel design rationale, see [`../kernels/`](../kernels/) — each
kernel doc includes a Design Decisions section.

## Planned design docs

- `backend-dtype-shape-contract.md` — the universal invariant every kernel
  in kuant preserves: numpy/cupy backend, dtype (with int → float64),
  shape (broadcasting + scalar-in/scalar-out), NaN propagation
- `naming-convention.md` — the no-underscore API rule (`bsput`, not
  `bs_put`; `belltest`, not `bell_test`) and the direction-in-name
  rule (only when call vs put math differs)
- `cpu-first-gpu-second.md` — why every kernel has a numpy fallback and
  the pattern for adding GPU acceleration incrementally
- `testing-policy.md` — the three-layer validation strategy (golden
  values, library reference match, cross-kernel identities) with
  concrete expected tolerances
- `dependency-management.md` — how sklearn / statsmodels are optional
  (lazy-imported at call time via `_require_*` helpers) so `import
  kuant` never pulls in heavy scientific-python deps
- `queueing-and-throttle.md` — hardware detection and adaptive
  chunking so consumer GPUs don't get overrun

Currently empty — populate as decisions accumulate.

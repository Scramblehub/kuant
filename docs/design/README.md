# kuant design docs

Cross-cutting architectural decisions that span multiple kernels or the
library as a whole.

For per-kernel rationale, see [`../kernels/`](../kernels/). Each kernel
doc has a Design Decisions section.

## Existing docs

- [`Validation_Additions.md`](Validation_Additions.md): validation gaps
  identified in a background audit. Priority-ranked list of missing
  input validators, runtime warnings, and message-polish items. Every
  Priority A gap has been addressed; the doc remains as the log.
- [`subpackage-scopes.md`](subpackage-scopes.md): working scopes for the
  five originally-empty subpackages (data, edgecases, signals, portfolio,
  text). Kernel list, first-three-cut recommendation, warnings, and
  sequencing per subpackage.
- [`plan-topology.md`](plan-topology.md): topology kernel plan.
- [`plan-options.md`](plan-options.md): options kernel plan.
- [`roadmap.md`](roadmap.md): what's next per subpackage.

## Planned docs

- `backend-dtype-shape-contract.md`: the universal invariant every
  kernel preserves (numpy/cupy backend, dtype with int-to-float64,
  shape broadcasting with scalar-in/scalar-out, NaN propagation).
- `naming-convention.md`: the no-underscore API rule (`bsput`, not
  `bs_put`) and the direction-in-name rule.
- `cpu-first-gpu-second.md`: why every kernel has a numpy fallback and
  the pattern for adding GPU acceleration incrementally.
- `testing-policy.md`: the three-layer validation strategy (golden
  values, library reference match, cross-kernel identities) with
  concrete expected tolerances.
- `dependency-management.md`: how scikit-learn, statsmodels, ripser,
  persim, pyarrow are optional. They import lazily inside the kernels
  that need them via `require_dep`, so `import kuant` stays cheap.
- `queueing-and-throttle.md`: hardware detection and adaptive chunking
  so consumer GPUs do not get overrun.

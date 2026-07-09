# kuant.causal

Causal-inference kernels for observational data. Each kernel targets a
distinct identification strategy: unit-level counterfactuals, exogenous
instruments, threshold-based natural experiments, and structural
discovery from joint distributions.

## What lives in the subpackage

Four independent estimators, no shared state:

1. [`synthcontrol`](synthcontrol.md). Abadie-Diamond-Hainmueller 2010
   synthetic control. Convex combination of donor units matched to a
   treated unit's pre-treatment path; post-treatment gap is the ATT.
2. [`iv`](iv.md). Two-stage least squares instrumental variables.
   Handles endogeneity when a valid instrument is available. Flags weak
   instruments per Staiger-Stock 1997.
3. [`rdd`](rdd.md). Sharp regression discontinuity. Local linear
   regression with triangular kernel on either side of a cutoff;
   framework from Imbens-Lemieux 2008.
4. [`pcalgo`](pcalgo.md). PC-algorithm skeleton phase. Fisher-Z partial
   correlation CI tests recover the undirected causal graph
   (Spirtes-Glymour-Scheines 2000, Kalisch-Buhlmann 2007).

## Convention

- All kernels return a frozen result dataclass with `.summary()` for
  quick printing.
- Sign conventions are documented per-kernel and match the standard
  applied-econometrics literature: `att` positive when treated
  outperforms synthetic; `tau` positive when RDD outcome jumps up at
  the cutoff; `beta` on the endogenous regressor as the causal
  coefficient.
- Numpy only. No cupy/JAX backends in v0.6.0 (batch 10). Inputs are
  coerced to `float64`; non-finite rows are dropped before estimation
  with a `KE-VAL-MIN-CLEAN` guard.
- Warnings surface identification concerns (weak instruments); errors
  surface data-shape and sample-size failures. Error codes appear
  verbatim in the raised message.

## When to reach for which kernel

| Design | Kernel | Key requirement |
| --- | --- | --- |
| One treated unit, many donors, aggregate outcome | `synthcontrol` | Pre-period fit possible from convex donor mix |
| Endogenous regressor, exogenous instrument | `iv` | `Cov(Z, U) = 0` and stage-1 F > 10 |
| Assignment jumps at a known threshold | `rdd` | Smoothness of `E[Y | X]` at the cutoff |
| Explore causal structure across many variables | `pcalgo` | Faithfulness + causal sufficiency |

## Related kernels

- [`kuant.stats`](../stats/README.md) supplies the rolling and
  regression primitives that any longitudinal use of these estimators
  will need upstream.
- [`kuant.qm.hmm`](../qm/hmm.md) is the natural companion when the
  treatment effect itself is regime-dependent: fit `synthcontrol` or
  `rdd` per HMM state instead of over the pooled sample.

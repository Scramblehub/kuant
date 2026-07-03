# Plan: kuant.options expansion + BS Greeks refactor

## Canonical structure (source of truth)

```text
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

## What went wrong

We shipped ALL BS Greeks (`bsputdelta`, `bscalldelta`, `bsputrho`,
`bscallrho`, `bsgamma`, `bsvega`) into `kuant.core`. That belongs in
`kuant.options` per the canonical spec. `kuant.core` should stay lean:
just BS pricing (`bsput`, `bscall`) as a math primitive, `normcdf`,
`normpdf`, plus other pure-math foundations.

## Refactor first, add second

### Step 1 — Reorganize BS Greeks: core → options

Move these files from `kuant/core/` to `kuant/options/`:

- `bsputdelta.py`
- `bscalldelta.py`
- `bsputrho.py`
- `bscallrho.py`
- `bsgamma.py`
- `bsvega.py`
- `_bs_common.py` (shared setup helper — moves with the Greeks that use it)

Update imports so each moved kernel does `from kuant.core import
bsput, bscall, normcdf, normpdf` (these stay in core). Update
`kuant/options/__init__.py` to re-export the Greeks; drop the Greeks
from `kuant/core/__init__.py`.

Move the corresponding docs from `docs/kernels/core/` to
`docs/kernels/options/`. Move the corresponding tests from
`tests/core/` to `tests/options/`. Update the paths and any
cross-references in the docs README.

Keep in `kuant.core`:

- `bsput`, `bscall` — BS pricing formulas (foundational math)
- `normcdf`, `normpdf` — normal distribution primitives

### Step 2 — After the refactor, add new kernels in `kuant.options`

All six new kernels go in `kuant.options` (BS options analytics).

#### Time-decay Greeks (theta)

Two direction-specific kernels like delta and rho:

- `bscalltheta`:

  ```math
  -S·e^(-q·T)·φ(d1)·σ/(2√T) + q·S·e^(-q·T)·Φ(d1) - r·K·e^(-r·T)·Φ(d2)
  ```

- `bsputtheta`:

  ```math
  -S·e^(-q·T)·φ(d1)·σ/(2√T) - q·S·e^(-q·T)·Φ(-d1) + r·K·e^(-r·T)·Φ(-d2)
  ```

Verify against a scipy-based reference. Cross-check by finite-
difference of `bsput`/`bscall` w.r.t. `T`.

#### Second-order-mixed Greek (charm, ∂δ/∂t)

Two direction-specific kernels:

- `bscallcharm`:

  ```math
  -e^(-q·T)·(q·Φ(d1) - φ(d1)·(2(r-q)T - d2·σ√T) / (2T·σ√T))
  ```

- `bsputcharm`:

  ```math
  -e^(-q·T)·(-q·Φ(-d1) - φ(d1)·(2(r-q)T - d2·σ√T) / (2T·σ√T))
  ```

Cross-check by finite-difference of `bsputdelta`/`bscalldelta` w.r.t.
`T`.

#### IV solver robustness

`impvolbisection` — bracketing-bisection IV solver, guaranteed to
converge on any monotonic price function. Fallback for the low-vega
tail where Newton (`impvol`) diverges. Ships next to `impvol` in
`kuant.options`. Two solvers side by side lets the user choose speed
(`impvol`) vs guaranteed convergence (`impvolbisection`).

## Design decisions to lock

### Theta sign convention

Textbook: negative for a long option (time decay costs the holder).
Match Bloomberg / most references.

### Theta units — annualized

Match the rest of the BS family: annualized (∂/∂T with T in years).
Users compute per-day as `theta / 365` at the call site. Document the
`/365` pattern in the doc.

### Separate `impvolbisection` function, not a `stable=True` flag

Cleaner namespace. Reader sees the algorithm name in the code
directly. `impvol` stays Newton-only; `impvolbisection` is the
bisection alternative.

### Three-layer cross-check pattern (matches the rest of the BS family)

1. Golden values from scipy-based reference (a few hand-picked
   `(S,K,T,r,σ,q)` tuples with expected output pasted into
   `pytest.parametrize`).
2. 1000-point random reference match at `atol=1e-10`.
3. Cross-kernel identity via finite-difference of the corresponding
   lower-order Greek (theta: FD of `bsput` w.r.t. `T`; charm: FD of
   `bsputdelta` w.r.t. `T`).

## Implementation order

1. Step 1 refactor — move BS Greeks from `core` to `options`.
   Mechanical file moves + import updates + doc/test path updates.
   Every existing test must pass before moving on.
2. `bscalltheta` + `bsputtheta` — one file `bstheta.py`.
3. `bscallcharm` + `bsputcharm` — one file `bscharm.py`.
4. `impvolbisection` — sits alongside `impvol` in `kuant.options`.

Docs go in `docs/kernels/options/`. Tests in `tests/options/`.

## `kuant.options` scope going forward

After tomorrow's work, `kuant.options` will hold:

- Existing: `impvol`
- Added: `bsputdelta`, `bscalldelta`, `bsputrho`, `bscallrho`,
  `bsgamma`, `bsvega`, `bscalltheta`, `bsputtheta`, `bscallcharm`,
  `bsputcharm`, `impvolbisection`
- Later: multi-leg spreads (`bullcallspread`, `strangleprice`, ...),
  chain filters (`optionchainbuild`), auto-hedging (`deltahedge`)

## Rough estimate

- Step 1 refactor — 1 hour (careful mechanical file moves, import
  updates, doc/test path fixes)
- Theta pair — 1 hour
- Charm pair — 1 hour
- `impvolbisection` — 1.5 hours

Total: 4.5 hours if all four ship. Realistic for tomorrow.

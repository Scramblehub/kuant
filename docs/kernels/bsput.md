# bsput — Black-Scholes European put pricer

## Purpose

Prices a European put option on a dividend-paying stock:

```math
d1 = [ln(S/K) + (r - q + σ²/2) · T] / (σ · √T)
d2 = d1 - σ · √T
put = K · e^(-r·T) · Φ(-d2) - S · e^(-q·T) · Φ(-d1)
```

Foundation for the M9 CDS-2008 analog put strategy and any future put-based
insurance sleeve. Uses `kuant.core.normcdf` internally — this is the **first
composed kernel** in kuant, validating that normcdf's contract holds up under
real load.

## Public API

```python
from kuant.core import bsput

price = bsput(S, K, T, r, sigma, q=0.0)
```

Any of the six inputs can be scalar or array; numpy broadcasting applies.

## Design decisions and rationale

### 1. Individual params instead of a struct

Matches scipy convention. Makes broadcasting explicit — passing an array of
strikes with scalar spot gives a strike curve; no struct-of-arrays gymnastics.

### 2. Backend detection across multiple inputs

If ANY input is a cupy array → whole computation runs on GPU. Reason: mixing
CPU and GPU inputs would force a round-trip either way, and the GPU trip is
usually the right call for research grids (millions of (S, K, T) tuples).

### 3. Broadcasting via `xp.broadcast_arrays`

Returns views, not copies — critical for large grids. Passing a `(1000, 1)`
strike vector and a `(1, 500)` tenor vector produces a `(1000, 500)` grid
without allocating a `(1000, 500, 6)` intermediate.

### 4. Dtype policy: derive from REQUIRED args only

Subtle correctness issue: the default `q = 0.0` is a Python float, which
`xp.asarray` turns into a **float64** array. If we included q in
`xp.result_type(...)`, a caller passing all-float32 inputs would silently be
promoted to float64.

Fix: compute `out_dtype` from `S, K, T, r, sigma` only, then coerce `q` with
that dtype. Now float32 in → float32 out, as promised.

### 5. Uniform-compute-then-mask pattern

We compute the analytic formula on the WHOLE grid, then use `xp.where` to
select normal vs edge-case cells. Alternative would be branching per-cell,
but on GPU that causes **warp divergence** — every thread in a warp has to
wait for the slowest branch. Uniform compute + mask stays SIMD-clean.

The "wasted" compute on edge cells is cheap because we substitute safe
values (`S=1, K=1, sigma=1, T=1`) before the log/divide, so no NaN or Inf
poisons the computation.

### 6. NaN propagation via `full_like(out, nan)` initialization

The natural way to handle NaN inputs: allocate output as NaN, then let each
branch's `where` selectively overwrite only the cells whose mask is True.
For NaN inputs, every mask evaluates False (NaN > 0 is False; NaN <= 0 is
False), so those cells stay NaN. Free, correct, no explicit NaN handling.

### 7. Edge cases layered by specificity

The overwrite order matters:

1. **Normal path** — analytic BS formula
2. **Deterministic** — `T ≤ 0` or `σ ≤ 0` → `max(K·e^(-r·T) - S·e^(-q·T), 0)`.
   Collapses correctly to `max(K - S, 0)` when T = 0.
3. **S = 0** — stock worthless, put worth `K · e^(-r·T)` (guaranteed exercise)
4. **K = 0** — put is worthless

Case B overrides Case A because S=0 with T>0 needs Case B, not Case A.

## Composition test: what this file tells us about normcdf

If normcdf silently returned float64 for float32 input, bsput would too.
If normcdf's backend detection broke on GPU, bsput would crash on GPU inputs.
Every bsput test is implicitly a normcdf integration test. This is why the
composition test (bsput uses normcdf) is a stronger validation than either
kernel's unit tests alone.

## Test coverage (23 tests)

1. **Golden values** — 5 hand-verified textbook cases (Hull Ch 15, ATM, OTM,
   ITM, with dividend). Values pulled from independent scipy computation.
2. **Reference match** — 1000 random parameter sets vs `scipy.stats.norm.cdf`
   directly (not our normcdf) → catches composition bugs.
3. **Broadcasting** — strike curve (1×N), full (strike × tenor) grid (N×M).
4. **Edge cases** — T=0 (expired), σ=0 (deterministic), S=0, K=0, NaN,
   float32 preservation, int promotion.
5. **Property tests** — monotonic in strike, monotonic in vol (vega > 0),
   non-negative, bounded by `K·e^(-r·T)`, put-call parity check.
6. **CPU==GPU parity** — 1000-element GPU output matches CPU to 1e-10.
7. **Backend promotion** — even one cupy input triggers full GPU compute.

## Real-world validation

Swap into `v8_bubble_sector_name_puts.py` where the BS pricer is currently
inline. Assert same CAGR on one M9 backtest → confirms no numerical drift.

## Performance notes

- CPU: composed of two `scipy.special.ndtr` calls per element, so ~2×
  normcdf's rate → ~100 ns/element
- GPU: two `cupyx.scipy.special.erf` calls + ~10 elementwise ops → ~2-3
  ns/element on RTX 4090
- Break-even vs CPU: ~50k elements

## Related kernels

- `kuant.core.normcdf` — called twice per bsput element
- `kuant.core.bscall` — call price, put-call parity partner
- `kuant.core.bsputdelta` — dP/dS, uses same normcdf pattern
- `kuant.core.bsputrho` — dP/dr, uses same normcdf pattern
- `kuant.core.bsgamma`, `kuant.core.bsvega` — put-call symmetric second-order Greeks

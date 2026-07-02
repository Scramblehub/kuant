# bsput — Black-Scholes European put pricer

## Purpose

Prices a European put on a dividend-paying stock:

```math
d1 = [ln(S/K) + (r - q + σ²/2) · T] / (σ · √T)
d2 = d1 - σ · √T
put = K · e^(-r·T) · Φ(-d2) - S · e^(-q·T) · Φ(-d1)
```

Foundation for the M9 CDS-2008 analog put strategy and any future put-based
insurance sleeve. First composed kernel in kuant — calls `normcdf` twice.

## Public API

```python
from kuant.core import bsput

price = bsput(S, K, T, r, sigma, q=0.0)
```

Any of the six inputs can be scalar or array; numpy broadcasting applies.

## Design decisions

### 1. Individual params instead of a struct

Matches scipy convention. Broadcasting stays explicit — array of strikes +
scalar spot gives a strike curve; no struct-of-arrays gymnastics.

### 2. Uses `_bs_common.prepare_bs` for setup

Backend detection, broadcasting, dtype policy, NaN init, and d1/d2/normal
mask all handled by the shared helper. See `_bs_common.py` for the flow.

### 3. Uniform-compute-then-mask pattern

Compute the analytic formula on the WHOLE grid, then use `xp.where` to
select normal vs edge-case cells. On GPU, branching per-cell causes warp
divergence — every thread waits for the slowest branch. Uniform compute +
mask stays SIMD-clean.

The "wasted" compute on edge cells is cheap because we substitute safe
placeholders (`S=1, K=1, sigma=1, T=1`) before the log/divide, so no NaN
or Inf poisons the pass.

### 4. Composition point

`normcdf(-d1)` and `normcdf(-d2)` are the two composition calls. Every
bsput test is implicitly a normcdf integration test. If normcdf silently
promoted dtype or broke on GPU, bsput would too.

## Edge cases

| Condition | Put |
| --- | --- |
| Normal | analytic |
| T=0 (expired) | max(K-S, 0) |
| σ=0, T>0 | max(K·e^(-r·T) - S·e^(-q·T), 0) |
| S=0 | K·e^(-r·T) (guaranteed exercise) |
| K=0 | 0 (worthless) |
| NaN | NaN |

Overwrite order matters: S=0 case overrides deterministic case because
`S=0, T>0` needs `K·e^(-r·T)`, not the intrinsic-discounted expression.

## Cross-check tests

- `test_matches_scipy` — 1000 random parameter sets vs scipy directly
- `test_put_call_parity_random` (in test_bscall.py) — C - P = S·e^(-q·T) - K·e^(-r·T) to 1e-12

## Test coverage (23 tests)

Golden values (5 textbook cases), scipy reference (1000 random),
broadcasting (strike curve + full grid), edge cases (T=0/σ=0/S=0/K=0/NaN,
float32, int promotion), property tests (monotonic in strike, monotonic
in vol, non-negative, bounded by K·e^(-r·T), put-call parity check),
CPU==GPU parity.

## Direct usage in kuant

M9 CDS-2008 analog put strategy — the pricer beneath the whole put sleeve.

## Related kernels

- `kuant.core.normcdf` — called twice per bsput element
- `kuant.core.bscall` — call price, put-call parity partner
- `kuant.core.bsputdelta` — dP/dS, uses same normcdf pattern
- `kuant.core.bsputrho` — dP/dr, uses same normcdf pattern
- `kuant.core.bsgamma`, `kuant.core.bsvega` — put-call symmetric second-order Greeks

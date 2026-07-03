# normpdf — Standard normal probability density function

## Purpose

`φ(x) = exp(-x²/2) / √(2π)`

The density of `N(0, 1)`. Simpler than `normcdf` — no special function
needed, just a fused exp/multiply. Every Greek that involves `∂²P/∂S²` or
`∂P/∂σ` routes through here: `bsgamma`, `bsvega`.

## Public API

```python
from kuant.core import normpdf
y = normpdf(x)
```

Same shape/dtype/backend contract as `normcdf`.

## Design decisions

### 1. No special-function library needed

`normcdf` needed `erf` (a library primitive, non-elementary). `normpdf` is
just `exp(-x²/2) / √(2π)` — one exp, one multiply. Native ufuncs on both
numpy and cupy. About 10× faster than `normcdf`.

### 2. Multiply-first ordering for FMA fusion

Written as `xp.exp(-0.5 * arr * arr) * _INV_SQRT_2PI` — the final multiply
by `1/√(2π)` follows `exp`, letting hardware fuse the trailing multiply
into a single instruction on modern GPUs. Alternative
`_INV_SQRT_2PI * xp.exp(...)` prevents fusion.

### 3. `±inf → 0` for free

`exp(-inf) = 0` naturally, so `φ(±inf) = 0` without an explicit branch.
Same for NaN (`exp(NaN) = NaN`). Both propagate through the ufunc chain.

### 4. Precomputed constant

`_INV_SQRT_2PI = 1/√(2π)` computed once at module load, not on every call.

## Edge cases

| Condition | Output |
| --- | --- |
| NaN | NaN |
| ±inf | 0.0 |
| int input | promoted to float64 |
| empty array | empty array |
| scalar | scalar |

## Cross-check tests

- `test_matches_scipy` — 10k random samples match `scipy.stats.norm.pdf` to 1e-15
- `test_symmetry` — φ(-x) == φ(x)
- `test_output_in_valid_range` — φ ∈ [0, 1/√(2π)]

## Test coverage (14 tests)

Golden values (x ∈ {0, ±1, ±2, 3, 5}), scipy match on 10k uniform,
NaN/±inf/empty/scalar/int/2D/float32, symmetry, output range, CPU==GPU
parity, backend preservation.

## Direct usage in kuant

Used by `bsgamma` and `bsvega` — every put-call-symmetric Greek that
involves the density (not the CDF).

## Related kernels

- `kuant.core.normcdf` — its integral
- `kuant.core.bsgamma`, `kuant.core.bsvega` — composed on normpdf

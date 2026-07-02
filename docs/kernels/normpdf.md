# normpdf — Standard normal probability density function

## Purpose

`φ(x) = exp(-x²/2) / √(2π)`

The density of `N(0, 1)`. Simpler than `normcdf` — no special function
needed, just a fused exp/multiply. Every Greek that involves `∂²P/∂S²`
or `∂P/∂σ` routes through this: `bsgamma`, `bsvega`, and the future
`bscallgamma` / `bscallvega`.

## Public API

```python
from kuant.core import normpdf
y = normpdf(x)
```

Same shape/dtype/backend contract as `normcdf`.

## Design decisions

### 1. No special-function library needed

`normcdf` needed `erf` (a library primitive because it's non-elementary).
`normpdf` is just `exp(-x²/2) / √(2π)` — one exp, one multiply. Native ufuncs
on both numpy and cupy. This makes it about 10× faster than `normcdf`.

### 2. Multiply-first ordering for FMA fusion

Written as `xp.exp(-0.5 * arr * arr) * _INV_SQRT_2PI` — the final multiply by
`1/√(2π)` is done after `exp`, so hardware can fuse the trailing `mul` into a
single instruction on modern GPUs. Alternative `_INV_SQRT_2PI * xp.exp(...)`
prevents fusion.

### 3. `±inf → 0` for free

`exp(-inf) = 0` naturally, so `φ(±inf) = 0` without an explicit branch. Same
for NaN (`exp(NaN) = NaN`). Both propagate through the ufunc chain — no
special-case code needed.

### 4. Precomputed constant

`_INV_SQRT_2PI = 1/√(2π)` computed once at module load, not on every call.

## Test coverage (14 tests)

- Golden values for x ∈ {0, ±1, ±2, 3, 5}
- scipy match on 10k uniform samples
- NaN / ±inf / empty / scalar / int / 2D / float32
- Symmetry: `φ(-x) == φ(x)`
- Output range: `[0, 1/√(2π)]`
- CPU==GPU parity

## Related

- `kuant.core.normcdf` — its integral
- `kuant.core.bsgamma`, `kuant.core.bsvega` — composed on this

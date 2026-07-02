# normcdf — Standard normal cumulative distribution function

## Purpose

`Φ(x) = P[Z ≤ x]` where `Z ~ N(0, 1)`.

Foundation kernel. Every BS pricer and rate-sensitivity Greek in kuant
routes through here: bsput, bscall, bsputdelta, bscalldelta, bsputrho,
bscallrho. Also used for z-score → p-value conversion and statistical tests.

## Public API

```python
from kuant.core import normcdf

result = normcdf(x)
```

- `x` — scalar / list / tuple / numpy array / cupy array
- Returns — same shape, dtype, and backend as input (int promoted to float64)

## Design decisions

### 1. Backend preservation (cupy in → cupy out)

Users control where their data lives. Silently transferring GPU arrays back
to CPU (or vice versa) would be surprising and expensive. We detect the
input backend and stay there.

### 2. dtype preservation (float32 in → float32 out)

Users choose their precision/speed trade-off. We do not silently promote or
demote. Exception: int input casts to float64 (numpy convention for CDF
functions).

### 3. Scalar in → scalar out

Numpy convention. Users passing a single value shouldn't have to `.item()`
the result. Enforced via `_prepare_input` returning a `was_scalar` flag.

### 4. NaN passthrough, ±inf saturation

- NaN in → NaN out
- +inf → 1.0
- -inf → 0.0

IEEE 754 conventions, matches scipy.

### 5. CPU path via scipy.special.ndtr

`scipy.special.ndtr` is the numerically stable reference. Both the numpy
path AND the tests use it as ground truth. We do NOT reimplement it.

### 6. GPU path via `cupyx.scipy.special.erf`

`Φ(x) = 0.5 + 0.5 · erf(x / √2)`.

`erf` calls CUDA's optimized implementation. Fast enough for typical usage.
A hand-written RawKernel (A&S 26.2.17 rational approximation) is kept as
commented reference code — enable it if profiling ever shows normcdf is a
bottleneck.

### 7. Throttle integration

Even though the library `erf` handles arbitrary sizes, we still respect the
throttle so a single call to `normcdf(billion_element_array)` can't
monopolize the GPU. The throttle returns a chunk-size int; we slice.

Chunking uses views (`flat[start:end]`) — no per-chunk allocation. Output
is pre-allocated once via `empty_like`.

### 8. Explicit sync at kernel boundaries

`cp.cuda.Stream.null.synchronize()` after each kernel launch, before
recording timing. Without this, `time.perf_counter()` measures kernel
launch, not kernel completion.

### 9. Memory-safety invariants

- Output allocated ONCE per call via `empty_like` — no allocation in the loop
- Chunk slices are views, not copies
- Throttle records only ints and floats — cannot hold array refs
- Explicit sync before recording timing
- No global state outside module-level constants and the singleton throttle

The autouse `_check_no_gpu_leak` fixture verifies no test leaves > 100 MB
of GPU memory allocated.

## Edge cases

| Condition | Output |
| --- | --- |
| NaN | NaN |
| +inf | 1.0 |
| -inf | 0.0 |
| int input | promoted to float64 |
| empty array | empty array |
| scalar | scalar (not 0-d array) |

## Cross-check tests

- `test_matches_scipy_ndtr_uniform` — 10k random samples match scipy to 1e-12
- `test_matches_scipy_ndtr_extreme` — wide-tail saturation match
- `test_gpu_matches_cpu` — GPU output bit-close (< 1e-12) to CPU output

## Test coverage (28 tests)

Golden values (9 reference points), scipy reference (uniform + extreme),
edge cases (NaN/±inf/empty/scalar/int/2D/3D/float32/list), property tests
(symmetry, monotonic, unit interval), CPU==GPU parity, backend preservation.

## Direct usage in kuant

Used by every BS pricer and rate-sensitivity Greek (bsput, bscall,
bsputdelta, bscalldelta, bsputrho, bscallrho). Foundation kernel.

## Related kernels

- `kuant.core.normpdf` — the density; normcdf is its integral
- `kuant.core.bsput`, `kuant.core.bscall` — call normcdf twice
- `kuant.core.bsputdelta`, `kuant.core.bscalldelta` — call normcdf once
- `kuant.core.bsputrho`, `kuant.core.bscallrho` — call normcdf once

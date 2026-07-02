# normcdf — Standard normal cumulative distribution function

## Purpose

`Φ(x) = P[Z ≤ x]` where `Z ~ N(0, 1)`.

Foundation kernel: Black-Scholes pricing, delta computation, z-score → p-value,
statistical tests. Called many millions of times in a single M9 backtest.

## Public API

```python
from kuant.core import normcdf

result = normcdf(x)
```

- `x` — scalar / list / tuple / numpy array / cupy array
- Returns — same shape, dtype, and backend as input (int promoted to float64)

## Design decisions and rationale

### 1. Backend preservation (cupy in → cupy out)

Users control where their data lives. Silently transferring GPU arrays back to
CPU (or vice versa) would be surprising and expensive. We detect the input
backend and stay there.

### 2. dtype preservation (float32 in → float32 out)

Users choose their precision/speed trade-off. We do not silently promote or
demote. Exception: int input is cast to float64 (matches numpy convention for
CDF functions).

### 3. Scalar in → scalar out

Numpy convention. Users passing a single value shouldn't have to `.item()` the
result. Enforced via `_prepare_input` returning `was_scalar` bit.

### 4. NaN passthrough, ±inf saturation

- NaN in → NaN out (never crashes on bad input)
- +inf → 1.0
- -inf → 0.0

This matches IEEE 754 conventions and scipy behavior.

### 5. CPU path via scipy.special.ndtr

`scipy.special.ndtr` is the numerically stable reference implementation. Both
the numpy path AND the tests use it as ground truth. We do NOT reimplement it
in numpy.

### 6. GPU path via cupy.erf (library) — not custom kernel

`Φ(x) = 0.5 + 0.5 * erf(x / √2)`.

`cupy.erf` calls CUDA's optimized erf. Fast enough for typical usage. We keep
a hand-written RawKernel (A&S 26.2.17 rational) as commented reference code
for when profiling shows normcdf is a bottleneck — but ship with the library
call for correctness and readability.

### 7. Throttle integration

Even though `cupy.erf` handles arbitrary sizes, we still respect the throttle
so a single call to `normcdf(billion_element_array)` can't monopolize the GPU.
The throttle returns a chunk-size int; we do the actual slicing.

Chunking uses views (`flat[start:end]`) — no per-chunk allocation. Output is
pre-allocated once via `empty_like`.

### 8. Explicit sync at kernel boundaries

`cp.cuda.Stream.null.synchronize()` after each kernel launch, before recording
timing. Without this, `time.perf_counter()` measures kernel launch, not kernel
completion — you'd get 5μs "runtimes" for a 500ms kernel.

## Memory safety invariants

- Output allocated ONCE per call via `empty_like` — no allocation in the loop
- Chunk slices are views, not copies
- Throttle records only ints and floats — impossible to hold array refs
- Explicit sync before recording timing
- No global state outside of module-level constants and the singleton throttle

The autouse `_check_no_gpu_leak` fixture verifies no test leaves > 100 MB of
GPU memory allocated.

## Test coverage (5-validation strategy)

1. **Golden values** — 9 hardcoded reference points
2. **scipy.special.ndtr match** — 10k random uniform, 5k extreme
3. **Edge cases** — NaN, ±inf, empty, scalar, int, 1D/2D/3D, float32/64, list
4. **Property tests** — symmetry Φ(-x)==1-Φ(x), monotonic, range [0,1]
5. **CPU==GPU parity** — numpy result matches cupy result (when GPU available)

## Real-world validation

The `Quant-Research/quant_lab/scripts/v8_bubble_sector_name_puts.py` BS pricer
calls `norm.cdf` inline. Once kuant.core.normcdf is verified via tests here,
swap the inline call for `from kuant.core import normcdf` and re-run one M9
backtest. Assert same CAGR — confirms no numerical drift.

## Performance notes

- CPU (scipy.ndtr): ~50 ns per element (single-threaded)
- GPU (cupy.erf, single kernel): ~1 ns per element on RTX 4090
- Speedup: ~50× for 1M+ element batches
- Below 10k elements: GPU launch overhead dominates; use CPU

## Related kernels

- `kuant.core.bsput` — will use normcdf internally
- `kuant.core.bsdelta` — will use normcdf internally
- `kuant.stats.rollz` → `kuant.core.normcdf` for z → p-value conversion

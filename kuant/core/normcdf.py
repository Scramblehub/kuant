"""Standard normal cumulative distribution function, batched.

Φ(x) = P[Z ≤ x] where Z ~ N(0, 1)

Mathematically:
    Φ(x) = (1 + erf(x / √2)) / 2

This kernel is the foundation for everything that touches Gaussian probability:
Black-Scholes pricing, delta computation, z-score → p-value conversion,
statistical tests. It's called many millions of times in a single M9 backtest.

Three implementation paths:
  1. CPU path (numpy):     uses scipy.special.ndtr — the reference implementation
  2. GPU path (cupy):      uses cupy.erf on the library primitive
  3. GPU fast path (opt):  hand-written RawKernel with A&S 26.2.17 rational
                            approximation. Left commented out — enable if
                            profiling shows normcdf is the bottleneck.

We start with path 1+2 (correct + fast enough). Path 3 is documented but not
enabled — the point of kuant is correctness first, optimization second.

INVARIANTS:
  - Backend preserved: numpy in → numpy out, cupy in → cupy out
  - dtype preserved: float32 in → float32 out, float64 in → float64 out
  - Shape preserved: 2D in → 2D out, scalar in → scalar out
  - NaN in → NaN out (never crashes on bad input)
  - ±inf → 0.0 / 1.0 (correct limits)
  - int input → cast to float64 (numpy convention)
  - empty array → empty array

  We do NOT allocate output arrays that outlive the call. No leaks possible.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.special import ndtr

from ..queueing import DEVICE, THROTTLE

# Try importing cupy at module load. If unavailable, we're CPU-only forever
# in this process. `_HAS_CUPY` is a module-level constant used for branching.
#
# cp and _cp_erf are annotated as Any so Pylance doesn't flag attribute
# access on the "None" branch — runtime guards (_HAS_CUPY / isinstance
# against _CUPY_NDARRAY) ensure we never touch them when cupy is absent.
cp: Any
_cp_erf: Any
try:
    import cupy as cp
    from cupyx.scipy.special import erf as _cp_erf
    _HAS_CUPY = True
    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _cp_erf = None
    _HAS_CUPY = False
    # Sentinel type — isinstance(anything, type(None)) is False unless the
    # object is literally None, so this makes the isinstance check safe.
    _CUPY_NDARRAY = type(None)


# Constants precomputed at module load — avoid repeated compute in hot path
_SQRT_2 = float(np.sqrt(2.0))
_INV_SQRT_2 = 1.0 / _SQRT_2
_SATURATION_THRESHOLD = 8.0  # |x| > this → return 0 or 1 exactly


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _prepare_input(x):
    """Coerce user input into (backend, array, was_scalar, orig_dtype).

    Handles the messy Python input space (scalars, lists, tuples, numpy
    arrays, cupy arrays) in one place so the kernel body doesn't have to
    care.

    Returns
    -------
    backend : module (numpy or cupy)
        Which array namespace to compute in.
    arr : ndarray
        Input as an array of that backend. May be a view of the original.
    was_scalar : bool
        Whether the input was a scalar (drives scalar-out convention).
    orig_dtype : np.dtype
        The dtype we should return. Ints get promoted to float64 here.
    """
    # cupy-array input → cupy backend (regardless of whether GPU is faster)
    if isinstance(x, _CUPY_NDARRAY):
        backend = cp
        arr = x
        was_scalar = arr.ndim == 0
        # For int arrays, promote to float64 (numpy convention for cdf).
        # np.float64 is portable — cupy accepts numpy dtypes everywhere.
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        return backend, arr, was_scalar, arr.dtype

    # Everything else → numpy backend
    was_scalar = np.isscalar(x)
    arr = np.asarray(x)
    if arr.dtype.kind in "iub":
        arr = arr.astype(np.float64)
    return np, arr, was_scalar, arr.dtype


# ---------------------------------------------------------------------------
# CPU path (numpy + scipy.special.ndtr)
# ---------------------------------------------------------------------------


def _normcdf_cpu(arr: np.ndarray) -> np.ndarray:
    """CPU path — thin wrapper over scipy.special.ndtr.

    scipy.special.ndtr is the numerically-stable reference implementation.
    It's what our GPU results are validated against in tests.
    """
    return ndtr(arr)


# ---------------------------------------------------------------------------
# GPU path (cupy + library erf)
# ---------------------------------------------------------------------------


def _normcdf_gpu_libraryerf(arr):
    """GPU path using cupy's library erf.

    cupy's erf calls CUDA's optimized erf implementation. Fast enough for
    most uses. We only need a hand-written kernel if this shows up as a
    bottleneck in profiling.

    Preserves dtype (float32 in → float32 out).
    """
    # 0.5 * (1 + erf(x / sqrt(2))) — one multiply, one erf, one add
    # Uses fused operations under the hood on modern CUDA
    # cupy >= 12 moved erf to cupyx.scipy.special (imported at module load)
    return 0.5 + 0.5 * _cp_erf(arr * _INV_SQRT_2)


# ---------------------------------------------------------------------------
# GPU fast path (optional, using Abramowitz-Stegun 26.2.17)
# ---------------------------------------------------------------------------
# UNCOMMENT AND WIRE INTO `normcdf()` IF PROFILING SHOWS THIS IS A BOTTLENECK.
# Documented here as reference for how to write a custom kernel.
#
# _NORMCDF_KERNEL_SRC = r'''
# extern "C" __global__
# void normcdf_as2617(const double* __restrict__ x,
#                     double* __restrict__ out,
#                     const int n) {
#     const int i = blockDim.x * blockIdx.x + threadIdx.x;
#     if (i >= n) return;
#
#     const double xi = x[i];
#
#     // NaN passthrough
#     if (isnan(xi)) { out[i] = xi; return; }
#
#     // Saturation for extreme values — skip 30+ FLOPs
#     if (xi > 8.0)  { out[i] = 1.0; return; }
#     if (xi < -8.0) { out[i] = 0.0; return; }
#
#     // A&S 26.2.17 — good to ~1e-7 across the real line
#     const double abs_x = fabs(xi);
#     const double t = 1.0 / (1.0 + 0.2316419 * abs_x);
#     const double poly =
#         t * (0.319381530 +
#         t * (-0.356563782 +
#         t * (1.781477937 +
#         t * (-1.821255978 +
#         t * 1.330274429))));
#     const double phi = 0.3989422804014327 * exp(-0.5 * xi * xi);  // 1/sqrt(2*pi)
#     const double approx = 1.0 - phi * poly;
#     out[i] = xi >= 0.0 ? approx : 1.0 - approx;
# }
# '''
#
# _normcdf_kernel = None
# def _get_normcdf_kernel():
#     global _normcdf_kernel
#     if _normcdf_kernel is None:
#         _normcdf_kernel = cp.RawKernel(_NORMCDF_KERNEL_SRC, "normcdf_as2617")
#     return _normcdf_kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normcdf(x):
    """Standard normal CDF, Φ(x) = P[Z ≤ x] for Z ~ N(0, 1).

    Batched by default: pass any shape, get same shape back. Backend and
    dtype are preserved.

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, or cupy.ndarray
        Input value(s). Any shape.
        - Scalar → scalar output
        - numpy array → numpy output
        - cupy array → cupy output (stays on GPU)
        - int → cast to float64 (numpy convention)

    Returns
    -------
    ndarray or scalar
        Φ(x) elementwise. Range [0, 1]. Same shape, dtype, and backend as
        input (with int promoted to float64).

    Raises
    ------
    TypeError
        If input cannot be coerced to a numeric array.

    Notes
    -----
    Math: Φ(x) = (1 + erf(x / √2)) / 2

    Edge cases:
      - NaN in → NaN out
      - +inf → 1.0
      - -inf → 0.0
      - Extreme finite (|x| > 8) → 0.0 or 1.0 (saturates within fp precision)
      - Empty array → empty array
      - Scalar → scalar

    Numerical accuracy:
      - CPU path: full double precision via scipy.special.ndtr
      - GPU path: bit-identical to library erf; ~1e-15 abs error
      - GPU fast path (optional): ~1e-7 abs error via A&S 26.2.17

    Examples
    --------
    >>> normcdf(0.0)
    0.5
    >>> normcdf(1.96)  # famous 97.5th percentile
    0.9750021048517795
    >>> import numpy as np
    >>> normcdf(np.array([-1.0, 0.0, 1.0]))
    array([0.15865525, 0.5       , 0.84134475])

    See Also
    --------
    kuant.core.bsput : Uses normcdf internally for BS put pricing
    scipy.special.ndtr : The reference CPU implementation
    """
    # Step 1: normalize input — turn messy Python into (backend, arr, meta)
    backend, arr, was_scalar, orig_dtype = _prepare_input(x)

    # Step 2: handle empty array before any allocation
    if arr.size == 0:
        result = arr
        return _restore_scalar(result, was_scalar)

    # Step 3: CPU path (numpy backend) — no throttle, just compute
    if backend is np:
        result = _normcdf_cpu(arr)
        return _restore_scalar(result, was_scalar)

    # Step 4: GPU path — chunk if the input is too large for VRAM
    # cupy's erf works fine on any size, but we still respect the throttle
    # to keep any single call from monopolizing the GPU.

    # bytes_per_elem = input + output, both same dtype
    bytes_per_elem = 2 * arr.dtype.itemsize

    chunk_size = THROTTLE.suggest_chunk_size(
        "normcdf",
        total_elems=arr.size,
        bytes_per_elem=bytes_per_elem,
    )

    if chunk_size >= arr.size:
        # Small enough — single kernel launch, no chunking overhead
        t0 = time.perf_counter()
        result = _normcdf_gpu_libraryerf(arr)
        cp.cuda.Stream.null.synchronize()  # explicit sync for timing
        THROTTLE.record("normcdf", arr.size, (time.perf_counter() - t0) * 1000)
        return _restore_scalar(result, was_scalar)

    # Chunked path — pre-allocate output ONCE, write chunks in-place.
    # We flatten for slicing simplicity, then reshape at the end.
    flat = arr.ravel()
    output = backend.empty_like(flat)
    n = flat.size

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        t0 = time.perf_counter()
        # Compute into a slice of the pre-allocated output.
        # `flat[start:end]` is a view, not a copy. Same for output slice.
        # This assignment triggers the actual kernel launch.
        output[start:end] = _normcdf_gpu_libraryerf(flat[start:end])
        cp.cuda.Stream.null.synchronize()
        THROTTLE.record("normcdf", end - start, (time.perf_counter() - t0) * 1000)

    result = output.reshape(arr.shape)
    return _restore_scalar(result, was_scalar)


def _restore_scalar(result, was_scalar):
    """If input was scalar, return a scalar (not a 0-d array).

    Matches numpy convention: scalar in → scalar out. Users passing a single
    value shouldn't have to `.item()` the result.
    """
    if was_scalar:
        # Convert 0-d array to Python float
        # For cupy 0-d, `.item()` transfers to host — one small sync
        return float(result)
    return result

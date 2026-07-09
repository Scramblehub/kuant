"""Standard normal cumulative distribution function, batched.

Φ(x) = P[Z ≤ x] where Z ~ N(0,1). Math: Φ(x) = (1 + erf(x/√2)) / 2.

Foundation kernel. Three implementation paths:
  1. CPU: scipy.special.ndtr (reference)
  2. GPU: cupyx.scipy.special.erf (library)
  3. Fast GPU: hand-written RawKernel (A&S 26.2.17). Kept as reference code;
     enable if profiling shows normcdf is a bottleneck.

Full design: docs/kernels/normcdf.md.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.special import ndtr

from ..queueing import THROTTLE

# cp/_cp_erf typed Any so Pylance doesn't complain on the None-branch;
# runtime guards (_HAS_CUPY / isinstance(_CUPY_NDARRAY)) ensure safety.
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
    # Sentinel: isinstance(x, type(None)) is False for any x != None.
    _CUPY_NDARRAY = type(None)


# Precomputed at module load — avoid recompute in hot path.
_SQRT_2 = float(np.sqrt(2.0))
_INV_SQRT_2 = 1.0 / _SQRT_2
_SATURATION_THRESHOLD = 8.0  # |x| > this → 0 or 1 exactly


def _prepare_input(x):
    """Coerce input into (backend, arr, was_scalar, orig_dtype).

    Handles scalar/list/tuple/numpy/cupy inputs uniformly. Int arrays
    promote to float64 (numpy convention for CDF). np.float64 is portable
    across numpy and cupy.
    """
    if isinstance(x, _CUPY_NDARRAY):
        arr = x
        was_scalar = arr.ndim == 0
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        return cp, arr, was_scalar, arr.dtype

    was_scalar = np.isscalar(x)
    arr = np.asarray(x)
    if arr.dtype.kind in "iub":
        arr = arr.astype(np.float64)
    return np, arr, was_scalar, arr.dtype


def _normcdf_cpu(arr: np.ndarray) -> np.ndarray:
    """CPU path — thin wrapper over scipy.special.ndtr (the reference)."""
    return ndtr(arr)


def _normcdf_gpu_libraryerf(arr):
    """GPU path via cupyx.scipy.special.erf. Preserves dtype."""
    # Multiply-first ordering lets hardware fuse the trailing multiply (FMA).
    return 0.5 + 0.5 * _cp_erf(arr * _INV_SQRT_2)


# ---------------------------------------------------------------------------
# GPU fast path (optional): Abramowitz-Stegun 26.2.17 rational approximation.
# UNCOMMENT AND WIRE INTO normcdf() IF PROFILING SHOWS THIS IS A BOTTLENECK.
# Reference implementation for how to write a custom CUDA kernel.
# ---------------------------------------------------------------------------
# _NORMCDF_KERNEL_SRC = r'''
# extern "C" __global__
# void normcdf_as2617(const double* __restrict__ x,
#                     double* __restrict__ out,
#                     const int n) {
#     const int i = blockDim.x * blockIdx.x + threadIdx.x;
#     if (i >= n) return;
#     const double xi = x[i];
#     if (isnan(xi)) { out[i] = xi; return; }
#     if (xi > 8.0)  { out[i] = 1.0; return; }
#     if (xi < -8.0) { out[i] = 0.0; return; }
#     const double abs_x = fabs(xi);
#     const double t = 1.0 / (1.0 + 0.2316419 * abs_x);
#     const double poly =
#         t * (0.319381530 +
#         t * (-0.356563782 +
#         t * (1.781477937 +
#         t * (-1.821255978 +
#         t * 1.330274429))));
#     const double phi = 0.3989422804014327 * exp(-0.5 * xi * xi);
#     const double approx = 1.0 - phi * poly;
#     out[i] = xi >= 0.0 ? approx : 1.0 - approx;
# }
# '''
#
# _normcdf_kernel = None
# def _get_normcdf_kernel():
#     global _normcdf_kernel
#     if _normcdf_kernel is None:
#         _normcdf_kernel = cp.RawKernel(_NORMCDF_KERNEL_SRC, 'normcdf_as2617')
#     return _normcdf_kernel


def normcdf(x):
    """Standard normal CDF, Φ(x) = P[Z ≤ x].

    Preserves backend (numpy/cupy), dtype (int → float64), shape, and
    scalar/array status. NaN → NaN; ±inf → 0/1.

    Examples
    --------
    >>> normcdf(0.0)
    0.5
    >>> normcdf(1.96)  # 97.5th percentile
    0.9750021048517795
    """
    backend, arr, was_scalar, orig_dtype = _prepare_input(x)

    if arr.size == 0:
        return _restore_scalar(arr, was_scalar)

    if backend is np:
        return _restore_scalar(_normcdf_cpu(arr), was_scalar)

    # GPU path — respect the throttle even though cupy erf handles any size.
    bytes_per_elem = 2 * arr.dtype.itemsize  # input + output
    chunk_size = THROTTLE.suggest_chunk_size(
        "normcdf",
        total_elems=arr.size,
        bytes_per_elem=bytes_per_elem,
    )

    if chunk_size >= arr.size:
        t0 = time.perf_counter()
        result = _normcdf_gpu_libraryerf(arr)
        cp.cuda.Stream.null.synchronize()  # sync for accurate timing
        THROTTLE.record("normcdf", arr.size, (time.perf_counter() - t0) * 1000)
        return _restore_scalar(result, was_scalar)

    # Chunked path — pre-allocate output ONCE; slices are views, not copies.
    flat = arr.ravel()
    output = backend.empty_like(flat)
    n = flat.size
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        t0 = time.perf_counter()
        output[start:end] = _normcdf_gpu_libraryerf(flat[start:end])
        cp.cuda.Stream.null.synchronize()
        THROTTLE.record("normcdf", end - start, (time.perf_counter() - t0) * 1000)

    return _restore_scalar(output.reshape(arr.shape), was_scalar)


def _restore_scalar(result, was_scalar):
    """Scalar in → scalar out (numpy convention). For cupy 0-d, .item() syncs."""
    return float(result) if was_scalar else result

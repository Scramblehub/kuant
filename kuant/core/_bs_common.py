"""Shared setup for Black-Scholes kernels — private, not exported.

Every kernel in the BS family (bsput, bsputdelta, bsgamma, bsvega,
bsputrho, bscall, ...) does the same 30 lines of setup:

  1. Detect backend (numpy vs cupy)
  2. Coerce required args to arrays
  3. Pick out_dtype from required args (avoid q's Python-scalar promotion)
  4. Coerce q with out_dtype
  5. Broadcast all six to common shape
  6. Cast to out_dtype
  7. Build a mask for "normal" cells (all inputs positive & finite)
  8. Substitute safe values in edge cells (avoid NaN/Inf poisoning)
  9. Compute d1, d2, sqrt_T
  10. Init out = full_like(NaN)

This module extracts that boilerplate into `prepare_bs()`, so each kernel
becomes:
    ctx = prepare_bs(S, K, T, r, sigma, q)
    <one-line formula using ctx.d1, ctx.d2, ctx.S_safe, ...>
    out = ctx.xp.where(ctx.normal, formula, ctx.out)
    <kernel-specific edge cases>
    return finalize(out)

Naming: fields with `_safe` suffix have edge-case cells replaced with 1.0
(safe placeholders that let the uniform-compute pass avoid NaN/Inf). Fields
without the suffix are the original broadcast, cast arrays — use those for
edge-case mask construction (S > 0, K > 0, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

cp: Any
try:
    import cupy as cp
    _HAS_CUPY = True
    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _HAS_CUPY = False
    _CUPY_NDARRAY = type(None)


def _detect_backend(*args) -> Any:
    """Pick numpy or cupy — any cupy input routes the whole call to GPU.

    Return type `Any` — both modules expose identical ufunc surfaces but
    share no static type.
    """
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


@dataclass
class BSContext:
    """Prepared context for a single BS kernel call. Everything a Greek
    formula needs, computed once and shared."""
    xp: Any
    out_dtype: Any
    # Original broadcast, cast inputs — use for edge-case mask construction
    S: Any
    K: Any
    T: Any
    r: Any
    sigma: Any
    q: Any
    # Safe versions (edge cells replaced with 1.0) for the uniform analytic pass
    S_safe: Any
    K_safe: Any
    T_safe: Any
    sigma_safe: Any
    # Precomputed intermediates
    sqrt_T: Any
    d1: Any
    d2: Any
    # Normal-cell mask (True where analytic formula applies)
    normal: Any
    # Pre-allocated output initialized to NaN
    out: Any


def prepare_bs(S, K, T, r, sigma, q=0.0) -> BSContext:
    """One-shot BS input preparation. See module docstring for the flow.

    Handles all the fiddly bits:
      - Backend detection across mixed numpy/cupy inputs
      - dtype policy (q's Python-scalar default doesn't force float64)
      - Broadcasting to common shape (views, no copy)
      - NaN-init output for free NaN propagation
      - Safe-value substitution in edge cells so log/divide don't poison
    """
    xp = _detect_backend(S, K, T, r, sigma, q)

    # Step 1: coerce required args (all but q); derive dtype from them only.
    S = xp.asarray(S)
    K = xp.asarray(K)
    T = xp.asarray(T)
    r = xp.asarray(r)
    sigma = xp.asarray(sigma)

    required_dtypes = [S.dtype, K.dtype, T.dtype, r.dtype, sigma.dtype]
    out_dtype = xp.result_type(*required_dtypes)
    if out_dtype.kind in "iub":
        out_dtype = xp.dtype(xp.float64)

    # Step 2: coerce q with the target dtype so its Python-scalar default 0.0
    # doesn't promote everything to float64.
    q = xp.asarray(q, dtype=out_dtype)

    # Step 3: broadcast all six to common shape (views).
    S, K, T, r, sigma, q = xp.broadcast_arrays(S, K, T, r, sigma, q)

    # Step 4: cast to out_dtype (may be no-op).
    S = S.astype(out_dtype, copy=False)
    K = K.astype(out_dtype, copy=False)
    T = T.astype(out_dtype, copy=False)
    r = r.astype(out_dtype, copy=False)
    sigma = sigma.astype(out_dtype, copy=False)

    # Step 5: init output as NaN. Any cell not overwritten (e.g. NaN inputs)
    # naturally propagates NaN.
    nan_val = xp.asarray(float("nan"), dtype=out_dtype)
    out = xp.full_like(S, nan_val)

    # Step 6: normal-cell mask.
    normal = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

    # Step 7: safe substitutions for uniform-compute-then-mask pattern.
    # Values in edge cells don't matter because `where(normal, ..., out)`
    # discards them; we just need them non-poisoning.
    one = xp.asarray(1.0, dtype=out_dtype)
    S_safe = xp.where(normal, S, one)
    K_safe = xp.where(normal, K, one)
    sigma_safe = xp.where(normal, sigma, one)
    T_safe = xp.where(normal, T, one)

    # Step 8: precompute d1, d2. Cheap even for kernels that only use d1.
    sqrt_T = xp.sqrt(T_safe)
    sigma_sqrt_T = sigma_safe * sqrt_T
    d1 = (xp.log(S_safe / K_safe) + (r - q + 0.5 * sigma_safe * sigma_safe) * T_safe) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    return BSContext(
        xp=xp, out_dtype=out_dtype,
        S=S, K=K, T=T, r=r, sigma=sigma, q=q,
        S_safe=S_safe, K_safe=K_safe, T_safe=T_safe, sigma_safe=sigma_safe,
        sqrt_T=sqrt_T, d1=d1, d2=d2,
        normal=normal, out=out,
    )


def finalize(out):
    """Convert 0-d output to Python scalar; return array otherwise.

    Matches numpy's scalar-in-scalar-out convention. Every kernel calls
    this as the final `return`.
    """
    if out.ndim == 0:
        return float(out)
    return out

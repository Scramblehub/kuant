'''Shared setup for Black-Scholes kernels — private, not exported.

Every BS kernel does the same setup:
  1. Detect backend (numpy/cupy)
  2. Coerce required args to arrays; derive out_dtype from them (avoid q's
     Python-scalar promotion)
  3. Broadcast all six inputs to common shape
  4. Init out = full_like(NaN) — free NaN propagation
  5. Substitute safe placeholder values in edge cells (avoid NaN/Inf poisoning)
  6. Compute d1, d2, sqrt_T

prepare_bs() returns a BSContext; each kernel writes ~20 lines of formula
on top. finalize() converts 0-d output to Python float.
'''
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
    '''Pick numpy or cupy — any cupy input routes the call to GPU.'''
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


@dataclass
class BSContext:
    '''Prepared context for a BS kernel call.

    Fields with `_safe` suffix have edge cells replaced with 1.0 (safe
    placeholders for the uniform analytic pass). Fields without the suffix
    are the original broadcast arrays — use those for edge-case masks
    (S > 0, K > 0, etc.).
    '''
    xp: Any
    out_dtype: Any
    S: Any
    K: Any
    T: Any
    r: Any
    sigma: Any
    q: Any
    S_safe: Any
    K_safe: Any
    T_safe: Any
    sigma_safe: Any
    sqrt_T: Any
    d1: Any
    d2: Any
    normal: Any
    out: Any


def prepare_bs(S, K, T, r, sigma, q=0.0) -> BSContext:
    '''One-shot BS input preparation. See module docstring for the flow.'''
    xp = _detect_backend(S, K, T, r, sigma, q)

    # Coerce required args; derive out_dtype from them ONLY so q's
    # Python-scalar default (float64) doesn't force everything to float64.
    S = xp.asarray(S)
    K = xp.asarray(K)
    T = xp.asarray(T)
    r = xp.asarray(r)
    sigma = xp.asarray(sigma)

    out_dtype = xp.result_type(S.dtype, K.dtype, T.dtype, r.dtype, sigma.dtype)
    if out_dtype.kind in 'iub':
        out_dtype = xp.dtype(xp.float64)

    q = xp.asarray(q, dtype=out_dtype)
    S, K, T, r, sigma, q = xp.broadcast_arrays(S, K, T, r, sigma, q)

    S = S.astype(out_dtype, copy=False)
    K = K.astype(out_dtype, copy=False)
    T = T.astype(out_dtype, copy=False)
    r = r.astype(out_dtype, copy=False)
    sigma = sigma.astype(out_dtype, copy=False)

    nan_val = xp.asarray(float('nan'), dtype=out_dtype)
    out = xp.full_like(S, nan_val)

    normal = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

    # Safe placeholders — values in edge cells don't matter since where()
    # discards them; they just need to not poison log/divide.
    one = xp.asarray(1.0, dtype=out_dtype)
    S_safe = xp.where(normal, S, one)
    K_safe = xp.where(normal, K, one)
    sigma_safe = xp.where(normal, sigma, one)
    T_safe = xp.where(normal, T, one)

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
    '''Scalar in → scalar out. Matches numpy convention.'''
    return float(out) if out.ndim == 0 else out

'''Standard normal probability density function, batched.

φ(x) = exp(-x²/2) / √(2π). Density of N(0,1).

Simpler than normcdf — no special function, just fused exp+multiply.
Used by bsgamma, bsvega.

Design: docs/kernels/normpdf.md.
'''
from __future__ import annotations

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


_INV_SQRT_2PI = float(1.0 / np.sqrt(2.0 * np.pi))


def _prepare_input(x):
    '''Coerce input into (backend, arr, was_scalar). Mirrors normcdf.'''
    if isinstance(x, _CUPY_NDARRAY):
        arr = x
        was_scalar = arr.ndim == 0
        if arr.dtype.kind in 'iub':
            arr = arr.astype(np.float64)
        return cp, arr, was_scalar

    was_scalar = np.isscalar(x)
    arr = np.asarray(x)
    if arr.dtype.kind in 'iub':
        arr = arr.astype(np.float64)
    return np, arr, was_scalar


def normpdf(x):
    '''Standard normal density, φ(x) = exp(-x²/2) / √(2π).

    Preserves backend/dtype/shape. Range [0, 1/√(2π)]. Symmetric.
    NaN → NaN; ±inf → 0.

    Examples
    --------
    >>> normpdf(0.0)
    0.3989422804014327
    >>> normpdf(1.0)
    0.24197072451914337
    '''
    xp: Any
    xp, arr, was_scalar = _prepare_input(x)

    if arr.size == 0:
        return arr

    # Multiply-first ordering enables FMA fusion on modern hardware.
    result = xp.exp(-0.5 * arr * arr) * _INV_SQRT_2PI

    return float(result) if was_scalar else result

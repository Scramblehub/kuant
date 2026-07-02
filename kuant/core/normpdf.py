"""Standard normal probability density function, batched.

phi(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)

The density function of the N(0, 1) distribution. Simpler than normcdf —
no special function needed, just a fused exp/multiply.

Used by bsgamma, bsvega, bscallgamma, bscallvega — every Greek that
involves d(price)/dS^2 or d(price)/dsigma routes through this.

INVARIANTS (matches normcdf):
  - Backend preserved
  - dtype preserved (int -> float64)
  - Scalar in -> scalar out
  - NaN in -> NaN out (via exp propagation)
  - +inf/-inf -> 0.0 (density vanishes at infinity)
  - Empty array -> empty array
"""
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


# 1 / sqrt(2*pi) — precomputed at module load
_INV_SQRT_2PI = float(1.0 / np.sqrt(2.0 * np.pi))


def _prepare_input(x):
    """Coerce input into (backend, arr, was_scalar). Mirrors normcdf helper.

    Note: uses np.float64 (not cp.float64) even for cupy arrays. numpy
    dtypes are portable — cupy accepts them everywhere. Avoids Pylance
    "attribute of None" complaints on the CPU-only fallback path where
    cp is None.
    """
    if isinstance(x, _CUPY_NDARRAY):
        arr = x
        was_scalar = arr.ndim == 0
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        return cp, arr, was_scalar

    was_scalar = np.isscalar(x)
    arr = np.asarray(x)
    if arr.dtype.kind in "iub":
        arr = arr.astype(np.float64)
    return np, arr, was_scalar


def normpdf(x):
    """Standard normal probability density, phi(x) = exp(-x^2/2) / sqrt(2*pi).

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, or cupy.ndarray

    Returns
    -------
    phi(x) : scalar or array
        Same shape, dtype, and backend as input. Range [0, 1/sqrt(2*pi)].

    Notes
    -----
    - phi(0) = 1/sqrt(2*pi) ~ 0.3989 (peak of the density)
    - phi(+-inf) = 0.0
    - Symmetric: phi(-x) == phi(x)

    Examples
    --------
    >>> normpdf(0.0)
    0.3989422804014327
    >>> normpdf(1.0)
    0.24197072451914337

    See Also
    --------
    kuant.core.normcdf : Standard normal CDF, its integral.
    """
    xp: Any
    xp, arr, was_scalar = _prepare_input(x)

    if arr.size == 0:
        return arr

    # phi(x) = exp(-x^2 / 2) / sqrt(2*pi)
    # Written as multiply-first so FMA fuses on modern hardware.
    result = xp.exp(-0.5 * arr * arr) * _INV_SQRT_2PI

    if was_scalar:
        return float(result)
    return result

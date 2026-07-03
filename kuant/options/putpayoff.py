'''European put expiry payoff, batched.

    payoff = max(K - S, 0)

Foundational building block for spread pricers, Monte Carlo terminal
value, and any expiry-time analytic. Not a Greek — this is the actual
exercise value at T.

Broadcasts S and K; backend detected from either input.

Design: docs/kernels/options/putpayoff.md.
'''
from __future__ import annotations

from typing import Any

import numpy as np

cp: Any
try:
    import cupy as cp
    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _detect_backend(*args) -> Any:
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def putpayoff(S, K):
    '''European put payoff at expiry: max(K - S, 0).

    Parameters
    ----------
    S : scalar or array
        Spot at expiry.
    K : scalar or array
        Strike.

    Returns
    -------
    payoff : scalar or array
        Shape follows broadcast of S and K. Backend follows either input
        being cupy; otherwise numpy.

    Examples
    --------
    >>> putpayoff(80.0, 100.0)
    20.0
    >>> putpayoff(120.0, 100.0)
    0.0
    '''
    xp = _detect_backend(S, K)
    S_arr = xp.asarray(S)
    K_arr = xp.asarray(K)

    out_dtype = xp.result_type(S_arr.dtype, K_arr.dtype)
    if out_dtype.kind in 'iub':
        out_dtype = xp.dtype(xp.float64)

    S_arr = S_arr.astype(out_dtype, copy=False)
    K_arr = K_arr.astype(out_dtype, copy=False)

    zero = xp.asarray(0.0, dtype=out_dtype)
    out = xp.maximum(K_arr - S_arr, zero)

    if out.ndim == 0:
        return float(out)
    return out

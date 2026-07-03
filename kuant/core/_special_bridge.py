'''Backend-aware bridge to scipy / cupyx special functions.

Design: kuant.core wants backend-preserving primitives, but implementing
special functions (gammaln, betainc, stdtrit, etc.) from scratch is a
large project. Instead, we bridge to:
    numpy input → scipy.special
    cupy input  → cupyx.scipy.special (if available)
                  → fallback: .get() to numpy, use scipy, then asarray back

The fallback path is slow (H↔D copy) but correct. Callers get consistent
API; performance-critical GPU work can be added later by porting kernels.

Private module — not exported from kuant.core.
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


def _is_cupy(a) -> bool:
    return cp is not None and isinstance(a, _CUPY_NDARRAY)


def _cupyx_special():
    '''Return cupyx.scipy.special module if available, else None.'''
    if cp is None:
        return None
    try:
        import cupyx.scipy.special as spx
        return spx
    except (ImportError, ModuleNotFoundError):
        return None


def _dispatch_unary(x, np_fn, spx_fn_name):
    '''Apply np_fn(x) or the cupyx equivalent depending on x's backend.'''
    if _is_cupy(x):
        spx = _cupyx_special()
        if spx is not None and hasattr(spx, spx_fn_name):
            return getattr(spx, spx_fn_name)(x)
        # Fallback: H↔D copy
        return cp.asarray(np_fn(x.get()))
    return np_fn(x)


def _dispatch_binary(a, b, np_fn, spx_fn_name):
    '''Apply np_fn(a, b) or cupyx equivalent.'''
    if _is_cupy(a) or _is_cupy(b):
        spx = _cupyx_special()
        if spx is not None and hasattr(spx, spx_fn_name):
            a_dev = cp.asarray(a)
            b_dev = cp.asarray(b)
            return getattr(spx, spx_fn_name)(a_dev, b_dev)
        # Fallback
        a_host = a.get() if _is_cupy(a) else np.asarray(a)
        b_host = b.get() if _is_cupy(b) else np.asarray(b)
        result = np_fn(a_host, b_host)
        return cp.asarray(result)
    return np_fn(a, b)


def _dispatch_ternary(a, b, c, np_fn, spx_fn_name):
    if _is_cupy(a) or _is_cupy(b) or _is_cupy(c):
        spx = _cupyx_special()
        if spx is not None and hasattr(spx, spx_fn_name):
            a_dev = cp.asarray(a)
            b_dev = cp.asarray(b)
            c_dev = cp.asarray(c)
            return getattr(spx, spx_fn_name)(a_dev, b_dev, c_dev)
        # Fallback
        a_host = a.get() if _is_cupy(a) else np.asarray(a)
        b_host = b.get() if _is_cupy(b) else np.asarray(b)
        c_host = c.get() if _is_cupy(c) else np.asarray(c)
        result = np_fn(a_host, b_host, c_host)
        return cp.asarray(result)
    return np_fn(a, b, c)


# --- Public helpers ---------------------------------------------------------

def gammaln(x):
    '''Log Γ(x), backend-preserving.'''
    from scipy.special import gammaln as np_gammaln
    return _dispatch_unary(x, np_gammaln, 'gammaln')


def betainc(a, b, x):
    '''Regularized incomplete beta I_x(a, b), backend-preserving.'''
    from scipy.special import betainc as np_betainc
    return _dispatch_ternary(a, b, x, np_betainc, 'betainc')


def stdtrit(df, p):
    '''Inverse Student-t CDF, numpy-only (no cupyx equivalent — H↔D fallback).'''
    from scipy.special import stdtrit as np_stdtrit
    return _dispatch_binary(df, p, np_stdtrit, 'stdtrit')

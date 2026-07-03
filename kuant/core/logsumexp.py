'''Numerically stable log-sum-exp, batched.

    logsumexp(x) = log(sum(exp(x)))

Computed as:
    m = max(x)
    logsumexp(x) = m + log(sum(exp(x - m)))

The subtraction of the max prevents overflow in exp(x). If m = -inf
(all x are -inf), returns -inf directly.

Used by: HMM forward-backward (kuant.qm), log-probability aggregation,
Bayesian model averaging, information-theoretic tests.

Design: docs/kernels/core/logsumexp.md.
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
    if cp is None:
        return np
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def logsumexp(x, axis=None, keepdims=False):
    '''Numerically stable log(sum(exp(x))).

    Parameters
    ----------
    x : array-like
        Values to log-sum-exp.
    axis : int or tuple of int, optional
        Axis or axes over which to reduce. Default reduces all axes.
    keepdims : bool, default False
        Preserve reduced dims as size-1.

    Returns
    -------
    scalar or array
        `log(sum(exp(x)))` computed without intermediate overflow.
        NaN in input propagates. All -inf in a reduced slice returns -inf.

    Examples
    --------
    >>> import numpy as np
    >>> abs(logsumexp(np.array([1000.0, 1000.0])) - (1000.0 + np.log(2))) < 1e-10
    True
    >>> logsumexp(np.array([-np.inf, -np.inf]))
    -inf
    '''
    xp = _detect_backend(x)
    x_arr = xp.asarray(x)

    if x_arr.dtype.kind in 'iub':
        x_arr = x_arr.astype(xp.float64)

    m = xp.max(x_arr, axis=axis, keepdims=True)

    # If m is -inf (all -inf), subtracting gives NaN; guard by treating -inf max as 0
    m_safe = xp.where(xp.isfinite(m), m, xp.asarray(0.0, dtype=m.dtype))
    shifted = x_arr - m_safe
    sum_exp = xp.sum(xp.exp(shifted), axis=axis, keepdims=True)
    # log(sum_exp) may be log(0) = -inf when all input is -inf. That is the
    # correct result (m is also -inf), but numpy raises a divide warning.
    with np.errstate(divide='ignore'):
        result = m + xp.log(sum_exp)

    # For elements where m was -inf, m + log(sum_exp) is -inf + (finite or nan) = -inf.
    # For elements where m was +inf (rare), it dominates and stays +inf.

    if not keepdims:
        result = xp.squeeze(result, axis=axis) if axis is not None else result.reshape(())

    if result.ndim == 0:
        return float(result)
    return result

"""Classify options by moneyness relative to spot.

Moneyness convention here: log-forward moneyness
    m = ln(K / F)   where F = S · e^((r-q)·T)

This is the natural quantity for BS math (d1, d2 are functions of m
scaled by σ√T). Log-moneyness has:
    m = 0  → ATM forward
    m > 0  → OTM call / ITM put
    m < 0  → ITM call / OTM put

Given per-option (S, K, T, r, q) and a set of bucket edges on m, this
kernel returns a bucket label per option.

Design: docs/kernels/options/moneynessbucket.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d

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


# Standard bucket labels used by many desks. Users can override.
DEFAULT_EDGES = np.array([-0.10, -0.03, 0.03, 0.10])
DEFAULT_LABELS = ("deep_ITM", "ITM", "ATM", "OTM", "deep_OTM")


def moneynessbucket(S, K, T, r, q=0.0, edges=None):
    """Assign each option a bucket index based on log-forward moneyness.

    Parameters
    ----------
    S, K, T, r : scalar or array (broadcast to common shape)
        Spot, strike, tenor, risk-free rate.
    q : scalar or array, default 0.0
        Continuous dividend yield.
    edges : 1D array, optional
        Bucket edges on log-forward moneyness m = ln(K/F).
        Default: [-0.10, -0.03, 0.03, 0.10] → 5 buckets.

    Returns
    -------
    bucket : int array (same shape as broadcast S,K,T,r,q)
        Integer bucket label 0..len(edges) where:
            0                → deep ITM call side (m < edges[0])
            1..len(edges)-1  → intermediate buckets
            len(edges)       → deep OTM call side (m >= edges[-1])

    Notes
    -----
    Convention is "call side": positive m = OTM CALL. For puts, flip
    the interpretation (deep_OTM in return means deep_ITM put).

    Examples
    --------
    >>> import numpy as np
    >>> S = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    >>> K = np.array([80.0, 97.0, 100.0, 103.0, 120.0])
    >>> # F = 100·e^(0.05) ≈ 105.13; K=100 is slightly ITM-forward
    >>> moneynessbucket(S, K, 1.0, 0.05)
    array([0, 1, 1, 2, 4])
    """
    xp = _detect_backend(S, K, T, r, q, edges)
    S_arr = xp.asarray(S)
    K_arr = xp.asarray(K)
    T_arr = xp.asarray(T)
    r_arr = xp.asarray(r)

    out_dtype = xp.result_type(S_arr.dtype, K_arr.dtype, T_arr.dtype, r_arr.dtype)
    if out_dtype.kind in "iub":
        out_dtype = xp.dtype(xp.float64)

    q_arr = xp.asarray(q, dtype=out_dtype)
    S_arr, K_arr, T_arr, r_arr, q_arr = xp.broadcast_arrays(S_arr, K_arr, T_arr, r_arr, q_arr)
    S_arr = S_arr.astype(out_dtype, copy=False)
    K_arr = K_arr.astype(out_dtype, copy=False)
    T_arr = T_arr.astype(out_dtype, copy=False)
    r_arr = r_arr.astype(out_dtype, copy=False)

    F = S_arr * xp.exp((r_arr - q_arr) * T_arr)
    m = xp.log(K_arr / F)

    if edges is None:
        edges_arr = xp.asarray(DEFAULT_EDGES, dtype=out_dtype)
    else:
        edges_arr = xp.asarray(edges, dtype=out_dtype)
        require_1d(edges_arr, "edges", kernel="moneynessbucket")

    # digitize: returns index in [0, len(edges)]
    return xp.digitize(m, edges_arr)

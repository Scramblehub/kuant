"""Implied volatility solver via vectorized bisection.

Complement to `impvol` (Newton-Raphson). Bisection is slower per
iteration but is BULLETPROOF: it cannot diverge, and it works even when
vega is tiny (deep OTM, near expiry) where Newton's step blows up.

Given market prices and (S, K, T, r, q), find the sigma such that:

    bsput(S, K, T, r, sigma, q) == price     (or bscall for calls)

Vectorized: all elements bisect simultaneously; converged elements are
frozen.

Failure modes → NaN:
  - Price outside no-arbitrage bounds
  - Not converged within max_iter (very rare with bisection)

Bisection convergence rate: ~log2((hi-lo)/tol) iterations.
For default lo=1e-6, hi=10, tol=1e-8: ~30 iterations.

Design: docs/kernels/options/impvolbisection.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantValueError

from ..core import bscall, bsput

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


_SIGMA_MIN_DEFAULT = 1e-6
_SIGMA_MAX_DEFAULT = 10.0


def _detect_backend(*args) -> Any:
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def impvolbisection(
    price,
    S,
    K,
    T,
    r,
    is_call=False,
    q=0.0,
    tol=1e-8,
    max_iter=100,
    sigma_lo=_SIGMA_MIN_DEFAULT,
    sigma_hi=_SIGMA_MAX_DEFAULT,
):
    """Vectorized bisection implied-vol solver.

    Parameters
    ----------
    price : scalar or array
        Observed option price(s). Must be in the no-arbitrage range for
        the corresponding (S, K, T, r, q).
    S, K, T, r : scalar or array
    is_call : bool, default False
    q : scalar or array, default 0.0
    tol : float, default 1e-8
        Convergence tolerance on the price residual.
    max_iter : int, default 100
        Iteration cap; ~30 is enough for the default bracket.
    sigma_lo, sigma_hi : float
        Search bracket. Default [1e-6, 10.0] covers 0.0001% to 1000%
        annualized vol.

    Returns
    -------
    sigma : scalar or array
        Implied volatility, shape follows broadcast of inputs. NaN where:
          - price < intrinsic (below arb bound)
          - price > upper bound (above arb bound)
          - target not bracketed by [sigma_lo, sigma_hi]

    Notes
    -----
    Bisection is slower than Newton but never diverges. Prefer this
    when accuracy on flat-vega tails (deep OTM, near expiry) matters
    more than speed. For fast typical-case solves, use `impvol`.

    Examples
    --------
    >>> import numpy as np
    >>> from kuant.core import bsput
    >>> sigma_true = 0.20
    >>> price = bsput(100.0, 100.0, 1.0, 0.05, sigma_true)
    >>> abs(impvolbisection(price, 100.0, 100.0, 1.0, 0.05) - sigma_true) < 1e-6
    True
    """
    require_positive(max_iter, "max_iter", kernel="impvolbisection", kind="int")
    require_positive(tol, "tol", kernel="impvolbisection")
    require_positive(sigma_lo, "sigma_lo", kernel="impvolbisection")
    require_positive(sigma_hi, "sigma_hi", kernel="impvolbisection")
    if float(sigma_lo) >= float(sigma_hi):
        raise KuantValueError(
            f"kuant.impvolbisection: 'sigma_lo' ({sigma_lo}) must be strictly "
            f"less than 'sigma_hi' ({sigma_hi}).  [KE-VAL-RANGE]\n"
            f"  → Fix: pass a non-degenerate bracket, e.g. sigma_lo=1e-6, "
            f"sigma_hi=10.0"
        )

    xp = _detect_backend(price, S, K, T, r, sigma_lo, sigma_hi)
    price_arr = xp.asarray(price)
    S_arr = xp.asarray(S)
    K_arr = xp.asarray(K)
    T_arr = xp.asarray(T)
    r_arr = xp.asarray(r)

    out_dtype = xp.result_type(price_arr.dtype, S_arr.dtype, K_arr.dtype, T_arr.dtype, r_arr.dtype)
    if out_dtype.kind in "iub":
        out_dtype = xp.dtype(xp.float64)

    q_arr = xp.asarray(q, dtype=out_dtype)
    price_arr, S_arr, K_arr, T_arr, r_arr, q_arr = xp.broadcast_arrays(
        price_arr, S_arr, K_arr, T_arr, r_arr, q_arr
    )
    price_arr = price_arr.astype(out_dtype, copy=True)
    S_arr = S_arr.astype(out_dtype, copy=False)
    K_arr = K_arr.astype(out_dtype, copy=False)
    T_arr = T_arr.astype(out_dtype, copy=False)
    r_arr = r_arr.astype(out_dtype, copy=False)

    lo = xp.full_like(S_arr, sigma_lo)
    hi = xp.full_like(S_arr, sigma_hi)

    price_fn = bscall if is_call else bsput

    # Endpoint prices (monotone increasing in sigma for calls and puts)
    price_lo = price_fn(S_arr, K_arr, T_arr, r_arr, lo, q_arr)
    price_hi = price_fn(S_arr, K_arr, T_arr, r_arr, hi, q_arr)

    # Bracketing check: price_lo <= target <= price_hi.
    bracketed = (price_lo <= price_arr + tol) & (price_arr <= price_hi + tol)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = price_fn(S_arr, K_arr, T_arr, r_arr, mid, q_arr) - price_arr
        # If f_mid > 0, mid is too high -> move hi down; else lo up.
        hi = xp.where(f_mid > 0, mid, hi)
        lo = xp.where(f_mid > 0, lo, mid)
        if bool(xp.all(hi - lo < tol)):
            break

    sigma = 0.5 * (lo + hi)
    nan_val = xp.asarray(float("nan"), dtype=out_dtype)
    sigma = xp.where(bracketed, sigma, nan_val)

    if sigma.ndim == 0:
        return float(sigma)
    return sigma

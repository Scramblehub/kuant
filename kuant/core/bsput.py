"""Black-Scholes European put pricer, batched.

put = K·e^(-r·T)·Φ(-d2) - S·e^(-q·T)·Φ(-d1)

First composed kernel in kuant — calls normcdf twice. Design: docs/kernels/bsput.md.
"""

from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bsput(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European put price.

    Parameters
    ----------
    S, K, T, r, sigma : scalar or array
        Spot, strike, tenor (years), rate (decimal), IV (decimal).
    q : scalar or array, default 0.0
        Continuous dividend yield (decimal).

    Examples
    --------
    >>> bsput(100.0, 100.0, 1.0, 0.05, 0.2)  # ATM 1y
    5.573526022256971
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    discount_r = xp.exp(-c.r * c.T_safe)
    discount_q = xp.exp(-c.q * c.T_safe)
    put_analytic = c.K_safe * discount_r * normcdf(-c.d2) - c.S_safe * discount_q * normcdf(-c.d1)
    out = xp.where(c.normal, put_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Deterministic (T<=0 or sigma<=0): PV of intrinsic. Collapses to
    # max(K-S, 0) at T=0.
    intrinsic_discounted = xp.maximum(c.K * xp.exp(-c.r * c.T) - c.S * xp.exp(-c.q * c.T), zero)
    deterministic = ((c.T <= 0) | (c.sigma <= 0)) & (c.S > 0) & (c.K > 0)
    out = xp.where(deterministic, intrinsic_discounted, out)

    # S=0 -> guaranteed exercise; overrides deterministic.
    S_zero = (c.S <= 0) & (c.K > 0)
    out = xp.where(S_zero, c.K * xp.exp(-c.r * c.T), out)

    # K=0 -> worthless.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

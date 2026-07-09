"""Black-Scholes European call pricer, batched.

call = S·e^(-q·T)·Φ(d1) - K·e^(-r·T)·Φ(d2)

Put-call parity: C - P = S·e^(-q·T) - K·e^(-r·T). Design: docs/kernels/bscall.md.
"""

from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bscall(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European call price. Non-negative.

    Examples
    --------
    >>> bscall(100.0, 100.0, 1.0, 0.05, 0.2)  # ATM 1y
    10.450583572185565
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    discount_r = xp.exp(-c.r * c.T_safe)
    discount_q = xp.exp(-c.q * c.T_safe)
    call_analytic = c.S_safe * discount_q * normcdf(c.d1) - c.K_safe * discount_r * normcdf(c.d2)
    out = xp.where(c.normal, call_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Deterministic (T<=0 or sigma<=0): PV of intrinsic. Collapses to
    # max(S-K, 0) at T=0.
    intrinsic_discounted = xp.maximum(c.S * xp.exp(-c.q * c.T) - c.K * xp.exp(-c.r * c.T), zero)
    deterministic = ((c.T <= 0) | (c.sigma <= 0)) & (c.S > 0) & (c.K > 0)
    out = xp.where(deterministic, intrinsic_discounted, out)

    # K=0 -> guaranteed exercise, unlimited payoff, worth S*exp(-q*T).
    K_zero = (c.K <= 0) & (c.S > 0)
    out = xp.where(K_zero, c.S * xp.exp(-c.q * c.T), out)

    # S=0 -> worthless. Must come last: K=0 AND S=0 -> worthless, not infinite.
    S_zero = c.S <= 0
    out = xp.where(S_zero, zero, out)

    return finalize(out)

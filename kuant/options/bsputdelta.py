"""Black-Scholes European put delta, batched.

delta = -e^(-q·T)·Φ(-d1)    range [-1, 0]

Directly powers M9 TP monitor's D70 rule. Design: docs/kernels/bsputdelta.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf


def bsputdelta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European put delta. Range [-1, 0].

    Examples
    --------
    >>> bsputdelta(100.0, 100.0, 1.0, 0.05, 0.2)  # ATM
    -0.3631693488243809
    >>> bsputdelta(100.0, 120.0, 1.0, 0.05, 0.2)  # deep ITM
    -0.7128083620948729
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    delta_analytic = -xp.exp(-c.q * c.T_safe) * normcdf(-c.d1)
    out = xp.where(c.normal, delta_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    neg_one_disc = -xp.exp(-c.q * c.T)  # guaranteed-exercise delta

    # Expired -> step at K.
    expired = (c.T <= 0) & (c.S > 0) & (c.K > 0)
    delta_expired = xp.where(c.K > c.S, xp.asarray(-1.0, dtype=c.out_dtype), zero)
    out = xp.where(expired, delta_expired, out)

    # Zero vol, T>0 -> deterministic step at forward break-even.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.K * xp.exp(-c.r * c.T) > c.S * xp.exp(-c.q * c.T)
    delta_zero_vol = xp.where(always_exercise, neg_one_disc, zero)
    out = xp.where(zero_vol, delta_zero_vol, out)

    # S=0 -> guaranteed exercise.
    S_zero = (c.S <= 0) & (c.K > 0)
    out = xp.where(S_zero, neg_one_disc, out)

    # K=0 -> never exercised.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

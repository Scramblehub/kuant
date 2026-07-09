"""Black-Scholes European call delta, batched.

delta = e^(-q·T)·Φ(d1)    range [0, 1]

Related to put delta: delta_call - delta_put = e^(-q·T) (parity identity).
Design: docs/kernels/bscalldelta.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf


def bscalldelta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European call delta. Range [0, 1].

    Examples
    --------
    >>> bscalldelta(100.0, 100.0, 1.0, 0.05, 0.20)
    0.6368306511756191
    >>> bscalldelta(100.0, 120.0, 1.0, 0.05, 0.20)  # OTM
    0.28719163790512714
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    delta_analytic = xp.exp(-c.q * c.T_safe) * normcdf(c.d1)
    out = xp.where(c.normal, delta_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    one_disc = xp.exp(-c.q * c.T)  # guaranteed-exercise delta

    # Expired -> step at K.
    expired = (c.T <= 0) & (c.S > 0) & (c.K > 0)
    delta_expired = xp.where(c.S > c.K, xp.asarray(1.0, dtype=c.out_dtype), zero)
    out = xp.where(expired, delta_expired, out)

    # Zero vol, T>0 -> deterministic step at forward break-even.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.S * xp.exp(-c.q * c.T) > c.K * xp.exp(-c.r * c.T)
    delta_zero_vol = xp.where(always_exercise, one_disc, zero)
    out = xp.where(zero_vol, delta_zero_vol, out)

    # K=0 -> guaranteed exercise (swapped vs put where this is at S=0).
    K_zero = (c.K <= 0) & (c.S > 0)
    out = xp.where(K_zero, one_disc, out)

    # S=0 -> worthless (swapped vs put where this is at K=0).
    S_zero = c.S <= 0
    out = xp.where(S_zero, zero, out)

    return finalize(out)

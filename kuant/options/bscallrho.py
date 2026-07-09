"""Black-Scholes European call rho, batched.

rho = T·K·e^(-r·T)·Φ(d2)    range [0, +inf)

Call-specific. Put rho: opposite sign, uses Φ(-d2). Design: docs/kernels/bscallrho.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf


def bscallrho(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European call rho. Range [0, +inf).

    For "rho per 1% rate change", divide by 100.

    Examples
    --------
    >>> bscallrho(100.0, 100.0, 1.0, 0.05, 0.20)
    53.232481545376345
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    rho_analytic = c.T_safe * c.K_safe * xp.exp(-c.r * c.T_safe) * normcdf(c.d2)
    out = xp.where(c.normal, rho_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Expired -> 0.
    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    # Zero vol, T>0 -> nonzero rho iff call exercises.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.S * xp.exp(-c.q * c.T) > c.K * xp.exp(-c.r * c.T)
    rho_zero_vol = xp.where(always_exercise, c.T * c.K * xp.exp(-c.r * c.T), zero)
    out = xp.where(zero_vol, rho_zero_vol, out)

    # S=0 -> call worthless -> 0.
    S_zero = c.S <= 0
    out = xp.where(S_zero, zero, out)

    # K=0 -> call worth S*exp(-q*T), no r-dependence -> 0.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

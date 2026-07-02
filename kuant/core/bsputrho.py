'''Black-Scholes European put rho, batched.

rho = -T·K·e^(-r·T)·Φ(-d2)    range (-inf, 0]

Put-specific. Call rho: opposite sign, uses Φ(d2). Design: docs/kernels/bsputrho.md.
'''
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bsputrho(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes European put rho. Range (-inf, 0].

    For "rho per 1% rate change", divide by 100.

    Examples
    --------
    >>> bsputrho(100.0, 100.0, 1.0, 0.05, 0.20)
    -41.890461500336574
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    rho_analytic = -c.T_safe * c.K_safe * xp.exp(-c.r * c.T_safe) * normcdf(-c.d2)
    out = xp.where(c.normal, rho_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Expired -> 0.
    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    # Zero vol, T>0 -> nonzero rho iff put exercises.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.K * xp.exp(-c.r * c.T) > c.S * xp.exp(-c.q * c.T)
    rho_zero_vol = xp.where(always_exercise, -c.T * c.K * xp.exp(-c.r * c.T), zero)
    out = xp.where(zero_vol, rho_zero_vol, out)

    # S=0 with T>0 -> guaranteed exercise -> -T*K*exp(-r*T).
    S_zero = (c.S <= 0) & (c.K > 0) & (c.T > 0)
    out = xp.where(S_zero, -c.T * c.K * xp.exp(-c.r * c.T), out)

    # K=0 -> 0.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

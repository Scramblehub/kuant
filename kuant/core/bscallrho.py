"""Black-Scholes European call rho, batched.

rho = +T * K * exp(-r*T) * Phi(d2)

Range: [0, +inf). Opposite sign from put rho (which is <= 0):
    put rho:  -T*K*exp(-r*T) * Phi(-d2)
    call rho: +T*K*exp(-r*T) * Phi(d2)

Uses kuant.core._bs_common.prepare_bs.

EDGE CASES:
  - T <= 0 (expired): 0 (no future to discount)
  - sigma <= 0, T > 0, always exercises: +T*K*exp(-r*T). Else 0.
  - S == 0: 0 (call worthless — no rate dependence to price)
  - K == 0: 0 (call worth S*exp(-q*T), no r dependence)
"""
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bscallrho(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European call rho.

    Range [0, +inf). For "rho per 1% rate change", divide by 100.

    Notes
    -----
    rho = T * K * exp(-r*T) * Phi(d2)

    Examples
    --------
    >>> bscallrho(100.0, 100.0, 1.0, 0.05, 0.20)
    53.232481545376345

    See Also
    --------
    kuant.core.bscall : bscallrho is its dr-derivative.
    kuant.core.bsputrho : Opposite sign, uses Phi(-d2) instead of Phi(d2).
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    rho_analytic = c.T_safe * c.K_safe * xp.exp(-c.r * c.T_safe) * normcdf(c.d2)
    out = xp.where(c.normal, rho_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Case A: expired -> 0.
    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    # Case B: zero vol, T > 0. Rho nonzero only if call exercises.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.S * xp.exp(-c.q * c.T) > c.K * xp.exp(-c.r * c.T)
    rho_zero_vol = xp.where(always_exercise, c.T * c.K * xp.exp(-c.r * c.T), zero)
    out = xp.where(zero_vol, rho_zero_vol, out)

    # Case C: S == 0 -> call worthless -> rho = 0.
    S_zero = (c.S <= 0)
    out = xp.where(S_zero, zero, out)

    # Case D: K == 0 -> call worth S*exp(-q*T), no r dependence -> rho = 0.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

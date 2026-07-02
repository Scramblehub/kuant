"""Black-Scholes European put rho, batched.

rho = -T * K * exp(-r*T) * Phi(-d2)

Range: (-inf, 0]. Put-specific (call rho has opposite sign and uses Phi(d2)).

Refactored to use kuant.core._bs_common.prepare_bs.

EDGE CASES:
  - T <= 0: 0 (no future to discount)
  - sigma <= 0, exercises: -T*K*exp(-r*T). Worthless: 0.
  - S == 0: -T*K*exp(-r*T) (guaranteed exercise)
  - K == 0: 0
"""
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bsputrho(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European put rho.

    Notes
    -----
    rho = -T * K * exp(-r*T) * Phi(-d2)

    Examples
    --------
    >>> bsputrho(100.0, 100.0, 1.0, 0.05, 0.20)
    -41.890461500336574

    See Also
    --------
    kuant.core.bsput : Rho is its dr-derivative.
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    rho_analytic = -c.T_safe * c.K_safe * xp.exp(-c.r * c.T_safe) * normcdf(-c.d2)
    out = xp.where(c.normal, rho_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Case A: expired -> 0.
    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    # Case B: zero vol, T > 0. Deterministic — nonzero rho if put exercises.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.K * xp.exp(-c.r * c.T) > c.S * xp.exp(-c.q * c.T)
    rho_zero_vol = xp.where(always_exercise, -c.T * c.K * xp.exp(-c.r * c.T), zero)
    out = xp.where(zero_vol, rho_zero_vol, out)

    # Case C: S == 0 with T > 0 -> guaranteed exercise -> -T*K*exp(-r*T).
    S_zero = (c.S <= 0) & (c.K > 0) & (c.T > 0)
    out = xp.where(S_zero, -c.T * c.K * xp.exp(-c.r * c.T), out)

    # Case D: K == 0 -> 0.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

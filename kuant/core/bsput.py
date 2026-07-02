"""Black-Scholes European put pricer, batched.

Prices a European put on a dividend-paying stock:

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put = K * exp(-r*T) * Phi(-d2) - S * exp(-q*T) * Phi(-d1)

First composed kernel in kuant — calls normcdf twice. If bsput tests pass,
normcdf's contract holds under real load.

Refactored to use kuant.core._bs_common.prepare_bs — see that module for the
shared setup logic. This file now focuses on the analytic formula + the
put-specific edge cases.

INVARIANTS (from _bs_common):
  - Backend preserved
  - dtype preserved (q default doesn't promote)
  - Broadcasting across all six inputs
  - NaN in -> NaN out (via full_like(nan) init)

EDGE CASES (put-specific):
  - T <= 0 (expired): max(K - S, 0)
  - sigma <= 0 (deterministic): max(K*exp(-r*T) - S*exp(-q*T), 0)
  - S == 0: put worth K*exp(-r*T) (guaranteed exercise)
  - K == 0: put worthless
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

    Returns
    -------
    put price(s) : scalar or array
        Same shape as broadcast of inputs. Backend and dtype preserved.

    Notes
    -----
    put = K * exp(-r*T) * Phi(-d2) - S * exp(-q*T) * Phi(-d1)

    Examples
    --------
    >>> bsput(100.0, 100.0, 1.0, 0.05, 0.2)  # ATM 1y
    5.573526022256971
    >>> import numpy as np
    >>> bsput(100.0, np.array([90, 100, 110]), 1.0, 0.05, 0.2)  # strike curve
    array([ 1.62410181,  5.57352602, 12.20232432])

    See Also
    --------
    kuant.core.normcdf : Called twice per bsput element.
    kuant.core.bsputdelta : dP/dS.
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    # Analytic put formula on the safe grid
    discount_r = xp.exp(-c.r * c.T_safe)
    discount_q = xp.exp(-c.q * c.T_safe)
    put_analytic = c.K_safe * discount_r * normcdf(-c.d2) - c.S_safe * discount_q * normcdf(-c.d1)
    out = xp.where(c.normal, put_analytic, c.out)

    # Edge cases, layered by specificity (later overrides earlier).
    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Case A: expired (T<=0) OR zero vol (sigma<=0) with S,K > 0.
    # For T=0: max(K - S, 0). For sigma=0, T>0: PV of deterministic intrinsic.
    #   max(K*exp(-r*T) - S*exp(-q*T), 0) — collapses to max(K-S, 0) at T=0.
    intrinsic_discounted = xp.maximum(c.K * xp.exp(-c.r * c.T) - c.S * xp.exp(-c.q * c.T), zero)
    deterministic = ((c.T <= 0) | (c.sigma <= 0)) & (c.S > 0) & (c.K > 0)
    out = xp.where(deterministic, intrinsic_discounted, out)

    # Case B: S == 0 -> put worth K*exp(-r*T) (guaranteed exercise).
    S_zero = (c.S <= 0) & (c.K > 0)
    out = xp.where(S_zero, c.K * xp.exp(-c.r * c.T), out)

    # Case C: K == 0 -> put worthless.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

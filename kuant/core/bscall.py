"""Black-Scholes European call pricer, batched.

Prices a European call on a dividend-paying stock:

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S * exp(-q*T) * Phi(d1) - K * exp(-r*T) * Phi(d2)

Note the sign differences from bsput:
  - Call:  +S*e^(-q*T)*Phi(d1)  -K*e^(-r*T)*Phi(d2)
  - Put:   -S*e^(-q*T)*Phi(-d1) +K*e^(-r*T)*Phi(-d2)

By put-call parity:  C - P = S*exp(-q*T) - K*exp(-r*T)

Uses kuant.core._bs_common.prepare_bs for setup.

EDGE CASES:
  - T <= 0 (expired): max(S - K, 0)
  - sigma <= 0 (deterministic forward): max(S*exp(-q*T) - K*exp(-r*T), 0)
  - S == 0: call worthless (can never be exercised profitably)
  - K == 0: call worth S*exp(-q*T) (guaranteed exercise, unlimited payoff)
"""
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bscall(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European call price.

    Parameters
    ----------
    S, K, T, r, sigma : scalar or array
        Spot, strike, tenor (years), rate (decimal), IV (decimal).
    q : scalar or array, default 0.0
        Continuous dividend yield (decimal).

    Returns
    -------
    call price(s) : scalar or array
        Non-negative. Same shape as broadcast of inputs.

    Notes
    -----
    call = S * exp(-q*T) * Phi(d1) - K * exp(-r*T) * Phi(d2)

    Examples
    --------
    >>> bscall(100.0, 100.0, 1.0, 0.05, 0.2)  # ATM 1y
    10.450583572185565

    See Also
    --------
    kuant.core.bsput : Put price. Related by put-call parity.
    kuant.core.bscalldelta : dC/dS.
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    # Analytic call formula on the safe grid
    discount_r = xp.exp(-c.r * c.T_safe)
    discount_q = xp.exp(-c.q * c.T_safe)
    call_analytic = c.S_safe * discount_q * normcdf(c.d1) - c.K_safe * discount_r * normcdf(c.d2)
    out = xp.where(c.normal, call_analytic, c.out)

    # Edge cases, layered by specificity.
    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Case A: expired (T<=0) OR zero vol (sigma<=0) with S,K > 0.
    # Both collapse to max(S*exp(-q*T) - K*exp(-r*T), 0), which reduces to
    # max(S-K, 0) at T=0.
    intrinsic_discounted = xp.maximum(c.S * xp.exp(-c.q * c.T) - c.K * xp.exp(-c.r * c.T), zero)
    deterministic = ((c.T <= 0) | (c.sigma <= 0)) & (c.S > 0) & (c.K > 0)
    out = xp.where(deterministic, intrinsic_discounted, out)

    # Case B: K == 0 -> guaranteed exercise, call worth S*exp(-q*T).
    # This is the CALL-specific case: strike zero means always exercise.
    # Overrides deterministic case for K=0 correctly.
    K_zero = (c.K <= 0) & (c.S > 0)
    out = xp.where(K_zero, c.S * xp.exp(-c.q * c.T), out)

    # Case C: S == 0 -> call worthless (nothing to buy).
    S_zero = c.S <= 0
    out = xp.where(S_zero, zero, out)

    return finalize(out)

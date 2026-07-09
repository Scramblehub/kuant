"""Black-Scholes gap option price, batched.

A gap option is a European vanilla whose TRIGGER strike (K_trigger)
determines whether the payoff is positive, but whose PAYOFF strike
(K_payoff) sets the size of the payoff. When K_payoff == K_trigger
the gap option reduces to a plain vanilla; when they differ, the
option can have a "gap" (discontinuous jump) at K_trigger.

Payoff at maturity:
    gap_call = (S_T - K_payoff) * indicator(S_T > K_trigger)
    gap_put  = (K_payoff - S_T) * indicator(S_T < K_trigger)

Closed form:
    d1 = (log(S/K_trigger) + (r - q + sigma^2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    gap_call = S*exp(-q*T)*N(d1) - K_payoff*exp(-r*T)*N(d2)
    gap_put  = K_payoff*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

Note that d1, d2 use K_TRIGGER (not K_payoff), while the discounted
K_payoff enters the second term. This is the same substitution that
produces the digital.

References
----------
Reiner & Rubinstein 1991, "Unscrambling the Binary Code." Haug 2007,
"The Complete Guide to Option Pricing Formulas", chapter 4.17.

Design: docs/kernels/options/gapprice.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf


def gapprice(S, K_trigger, K_payoff, T, r, sigma, q=0.0, *, is_call=True):
    """Black-Scholes gap option price.

    Parameters
    ----------
    S, T, r, sigma, q : scalar or array
        Standard Black-Scholes inputs.
    K_trigger : scalar or array
        Trigger strike: payoff activates when `S_T > K_trigger` (call)
        or `S_T < K_trigger` (put).
    K_payoff : scalar or array
        Payoff strike: the amount subtracted / added when payoff fires.
    is_call : bool, default True

    Returns
    -------
    scalar or array

    Examples
    --------
    Vanilla-equivalent when triggers match:

    >>> round(gapprice(100.0, 100.0, 100.0, 1.0, 0.05, 0.20), 4)
    10.4506

    Nonzero gap example (trigger 100, payoff 90):

    >>> round(gapprice(100.0, 100.0, 90.0, 1.0, 0.05, 0.20), 4)
    15.7738
    """
    # prepare_bs uses K_trigger for the d1/d2 setup (the "strike" that
    # determines the probability of triggering).
    c = prepare_bs(S, K_trigger, T, r, sigma, q)
    xp = c.xp

    K_payoff_arr = xp.asarray(K_payoff, dtype=c.out_dtype)
    disc_r = xp.exp(-c.r * c.T_safe)
    disc_q = xp.exp(-c.q * c.T_safe)
    if is_call:
        price = c.S_safe * disc_q * normcdf(c.d1) - K_payoff_arr * disc_r * normcdf(c.d2)
    else:
        price = K_payoff_arr * disc_r * normcdf(-c.d2) - c.S_safe * disc_q * normcdf(-c.d1)

    c.out = xp.where(
        (c.T > 0) & (c.sigma > 0) & (c.S > 0) & (c.K > 0),
        price,
        c.out,
    )
    return finalize(c.out)


__all__ = ["gapprice"]

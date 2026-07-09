"""Black-Scholes European digital (cash-or-nothing binary) option price, batched.

A digital call pays a fixed cash amount `cash` if the underlying finishes
above the strike at maturity, zero otherwise. Put analogue pays if below.

Closed form (under Black-Scholes, continuous dividend yield q):
    digital_call = cash * exp(-r*T) * N(d2)
    digital_put  = cash * exp(-r*T) * N(-d2)

where d2 = (log(S/K) + (r - q - sigma^2/2)*T) / (sigma*sqrt(T)).

This is the risk-neutral probability of finishing ITM, discounted, times
the fixed payout. The step-function payoff makes the Greeks singular
at the strike near expiry; a smoothed-delta variant may follow in a
later release.

References
----------
Reiner & Rubinstein 1991, "Unscrambling the Binary Code." Haug 2007,
"The Complete Guide to Option Pricing Formulas", chapter 4.19.

Design: docs/kernels/options/digitalprice.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf


def digitalprice(S, K, T, r, sigma, q=0.0, *, cash=1.0, is_call=True):
    """Black-Scholes European digital (cash-or-nothing) price.

    Parameters
    ----------
    S, K, T, r, sigma, q : scalar or array
        Standard Black-Scholes inputs; q is continuous dividend yield.
    cash : scalar or array, default 1.0
        Cash payout on knock-in at maturity.
    is_call : bool, default True
        `True` for a cash-or-nothing call (pays if S_T > K), `False`
        for the put analogue (pays if S_T < K).

    Returns
    -------
    scalar or array
        Price of the digital option, same shape as broadcast inputs.

    Examples
    --------
    >>> round(digitalprice(100.0, 100.0, 1.0, 0.05, 0.20, cash=1.0), 4)
    0.5323
    >>> round(digitalprice(100.0, 100.0, 1.0, 0.05, 0.20, cash=1.0, is_call=False), 4)
    0.4189
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    cash_arr = xp.asarray(cash, dtype=c.out_dtype)
    disc = xp.exp(-c.r * c.T_safe)
    if is_call:
        prob = normcdf(c.d2)
    else:
        prob = normcdf(-c.d2)
    price = cash_arr * disc * prob

    c.out = xp.where(
        (c.T > 0) & (c.sigma > 0) & (c.S > 0) & (c.K > 0),
        price,
        c.out,
    )
    return finalize(c.out)


__all__ = ["digitalprice"]

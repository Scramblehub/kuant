"""Black-Scholes European option vega, batched. Same for calls and puts.

    vega = S * exp(-q*T) * phi(d1) * sqrt(T)

Put-call symmetry: vega has the same value for a call and a put with
identical (S, K, T, r, sigma, q). No direction argument needed.

Units: vega is per unit sigma (decimal). For "vega per 1% IV change", /100.

Range [0, +inf).

Refactored to use kuant.core._bs_common.prepare_bs.

EDGE CASES: all collapse to vega = 0.
"""
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normpdf import normpdf


def bsvega(S, K, T, r, sigma, q=0.0):
    """Black-Scholes vega. Same for calls and puts (put-call symmetric).

    Notes
    -----
    vega = S * exp(-q*T) * phi(d1) * sqrt(T)

    Examples
    --------
    >>> bsvega(100.0, 100.0, 1.0, 0.05, 0.20)
    37.52403469169379

    See Also
    --------
    kuant.core.bsput, kuant.core.bscall : Vega is their dsigma-derivative.
    kuant.core.normpdf : Called once per vega element.
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    vega_analytic = c.S_safe * xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * c.sqrt_T
    out = xp.where(c.normal, vega_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

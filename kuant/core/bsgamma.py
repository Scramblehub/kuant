"""Black-Scholes European option gamma, batched. Same for calls and puts.

    gamma = exp(-q*T) * phi(d1) / (S * sigma * sqrt(T))

Put-call symmetry: gamma has the same value for a call and a put with
identical (S, K, T, r, sigma, q). No direction argument needed.

Range [0, +inf).

Refactored to use kuant.core._bs_common.prepare_bs.

EDGE CASES: all collapse to gamma = 0.
"""
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normpdf import normpdf


def bsgamma(S, K, T, r, sigma, q=0.0):
    """Black-Scholes gamma. Same for calls and puts (put-call symmetric).

    Notes
    -----
    gamma = exp(-q*T) * phi(d1) / (S * sigma * sqrt(T))

    Examples
    --------
    >>> bsgamma(100.0, 100.0, 1.0, 0.05, 0.20)
    0.018762017345846895

    See Also
    --------
    kuant.core.bsputdelta, kuant.core.bscalldelta : Gamma is their dS-derivative.
    kuant.core.normpdf : Called once per gamma element.
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    gamma_analytic = xp.exp(-c.q * c.T_safe) * normpdf(c.d1) / (c.S_safe * c.sigma_safe * c.sqrt_T)
    out = xp.where(c.normal, gamma_analytic, c.out)

    # All edges collapse to zero.
    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

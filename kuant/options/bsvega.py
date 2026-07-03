'''Black-Scholes vega, batched. Put-call symmetric — one kernel, both directions.

vega = S·e^(-q·T)·φ(d1)·√T    range [0, +inf)

Units: per unit sigma (decimal). For "per 1% IV change", divide by 100.
Design: docs/kernels/bsvega.md.
'''
from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normpdf import normpdf


def bsvega(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes vega. Same for calls and puts.

    Examples
    --------
    >>> bsvega(100.0, 100.0, 1.0, 0.05, 0.20)
    37.52403469169379
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    vega_analytic = c.S_safe * xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * c.sqrt_T
    out = xp.where(c.normal, vega_analytic, c.out)

    # All edges collapse to 0.
    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

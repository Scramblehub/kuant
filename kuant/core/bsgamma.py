'''Black-Scholes gamma, batched. Put-call symmetric — one kernel, both directions.

gamma = e^(-q·T)·φ(d1) / (S·σ·√T)    range [0, +inf)

Design: docs/kernels/bsgamma.md.
'''
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normpdf import normpdf


def bsgamma(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes gamma. Same for calls and puts.

    Examples
    --------
    >>> bsgamma(100.0, 100.0, 1.0, 0.05, 0.20)
    0.018762017345846895
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    gamma_analytic = xp.exp(-c.q * c.T_safe) * normpdf(c.d1) / (c.S_safe * c.sigma_safe * c.sqrt_T)
    out = xp.where(c.normal, gamma_analytic, c.out)

    # All edges collapse to 0.
    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

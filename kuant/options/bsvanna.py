'''Black-Scholes vanna, batched. Put-call symmetric.

vanna = ∂Delta/∂σ = ∂Vega/∂S
      = -e^(-qT) · φ(d1) · d2 / σ

Units: per unit σ (decimal). For "per 1% IV change" divide by 100.

Vanna is the same for calls and puts because delta_call - delta_put =
e^(-qT) has no σ-dependence. Same reason vega is put-call symmetric.

Design: docs/kernels/options/bsvanna.md.
'''
from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normpdf import normpdf


def bsvanna(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes vanna. Same for calls and puts.

    Examples
    --------
    >>> bsvanna(100.0, 100.0, 1.0, 0.05, 0.20)
    -19.7601517886541
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    vanna_analytic = -xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * c.d2 / c.sigma_safe
    out = xp.where(c.normal, vanna_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

'''Black-Scholes zomma, batched. Put-call symmetric.

zomma = ∂Gamma/∂σ = ∂³Price/∂S²∂σ
      = Gamma · (d1·d2 - 1) / σ
      = e^(-qT)·φ(d1) / (S·σ·√T) · (d1·d2 - 1) / σ

Units: per unit σ.

Zomma measures how gamma responds to σ moves; useful for hedge
rebalancing when IV shifts across the vol surface.

Put-call symmetric because gamma is put-call symmetric.

Design: docs/kernels/options/bszomma.md.
'''
from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normpdf import normpdf


def bszomma(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes zomma. Same for calls and puts.

    Examples
    --------
    >>> bszomma(100.0, 100.0, 1.0, 0.05, 0.20)
    -0.09880075894327053
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    sigma_sqrt_T = c.sigma_safe * c.sqrt_T
    gamma = xp.exp(-c.q * c.T_safe) * normpdf(c.d1) / (c.S_safe * sigma_sqrt_T)
    zomma_analytic = gamma * (c.d1 * c.d2 - 1.0) / c.sigma_safe
    out = xp.where(c.normal, zomma_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

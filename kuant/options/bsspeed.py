'''Black-Scholes speed, batched. Put-call symmetric.

speed = ∂Gamma/∂S = ∂³Price/∂S³
      = -Gamma / S · (d1 / (σ√T) + 1)
      = -e^(-qT)·φ(d1) / (S²·σ·√T) · (d1/(σ√T) + 1)

Units: per unit spot. Speed measures how gamma changes as spot moves;
important for hedging near strikes and for pin-risk near expiry.

Put-call symmetric because gamma is put-call symmetric.

Design: docs/kernels/options/bsspeed.md.
'''
from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normpdf import normpdf


def bsspeed(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes speed. Same for calls and puts.

    Examples
    --------
    >>> bsspeed(100.0, 100.0, 1.0, 0.05, 0.20)
    -0.0002187519921867094
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    sigma_sqrt_T = c.sigma_safe * c.sqrt_T
    gamma = xp.exp(-c.q * c.T_safe) * normpdf(c.d1) / (c.S_safe * sigma_sqrt_T)
    speed_analytic = -gamma / c.S_safe * (c.d1 / sigma_sqrt_T + 1.0)
    out = xp.where(c.normal, speed_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

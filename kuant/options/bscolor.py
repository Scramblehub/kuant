'''Black-Scholes color, batched. Put-call symmetric.

color = ∂Gamma/∂T = ∂³Price/∂S²∂T
      = -e^(-qT) · φ(d1) / (2·S·T·σ·√T)
        · [ 2·q·T + 1 + (2(r-q)T - d2·σ·√T) · d1 / (σ·√T) ]

Sign convention: like charm, "color" traditionally means the rate at
which gamma DECAYS with the passage of calendar time forward
(color = -∂Gamma/∂τ where τ is time-to-expiry). Here we return
∂Gamma/∂T directly so a positive value means gamma grows with more
time-to-expiry (typical for OTM options).

Put-call symmetric because gamma is.

Design: docs/kernels/options/bscolor.md.
'''
from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normpdf import normpdf


def bscolor(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes color = ∂Gamma/∂T. Same for calls and puts.

    Sign: ∂Gamma/∂T. Divide by 252 for per-trading-day gamma change.

    Examples
    --------
    >>> bscolor(100.0, 100.0, 1.0, 0.05, 0.20)
    0.00787527971924437
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    sigma_sqrt_T = c.sigma_safe * c.sqrt_T
    inner = 2.0 * c.q * c.T_safe + 1.0 + (2.0 * (c.r - c.q) * c.T_safe - c.d2 * sigma_sqrt_T) * c.d1 / sigma_sqrt_T
    prefactor = -xp.exp(-c.q * c.T_safe) * normpdf(c.d1) / (2.0 * c.S_safe * c.T_safe * sigma_sqrt_T)
    color_analytic = prefactor * inner
    out = xp.where(c.normal, color_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

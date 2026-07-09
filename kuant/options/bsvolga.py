"""Black-Scholes volga (vomma), batched. Put-call symmetric.

volga = ∂²Price/∂σ² = ∂Vega/∂σ
      = Vega · d1·d2 / σ
      = S·e^(-qT)·φ(d1)·√T · d1·d2 / σ

Units: per unit σ². For "per 1% IV change squared" divide by 10000.

Volga measures the convexity of price w.r.t. σ — critical for vol-of-vol
exposure and pricing volatility products.

Put-call symmetric because the put-call parity difference is σ-independent.

Design: docs/kernels/options/bsvolga.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normpdf import normpdf


def bsvolga(S, K, T, r, sigma, q=0.0):
    """Black-Scholes volga (vomma). Same for calls and puts.

    Examples
    --------
    >>> bsvolga(100.0, 100.0, 1.0, 0.05, 0.20)
    8.130197126032013
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    vega = c.S_safe * xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * c.sqrt_T
    volga_analytic = vega * c.d1 * c.d2 / c.sigma_safe
    out = xp.where(c.normal, volga_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    edge = (~c.normal) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(edge, zero, out)

    return finalize(out)

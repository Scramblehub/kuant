'''Black-Scholes European put charm, batched.

charm_put = -∂Delta_put/∂T                   (delta decay per year of
                                              calendar time passing)

Closed form:
    charm_put = -q·e^(-q·T)·Φ(-d1)
                - e^(-q·T)·φ(d1) · [ (2(r-q)T - d2·σ·√T) / (2·T·σ·√T) ]

Sign convention matches bscallcharm — the rate at which put delta
BLEEDS as calendar time passes forward.

Put charm relates to call charm via parity: differentiating
delta_call - delta_put = e^(-q·T) with respect to T gives
charm_call - charm_put = -q·e^(-q·T) (rearranged for the -dDelta/dT
convention).

Design: docs/kernels/options/bsputcharm.md.
'''
from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf
from ..core.normpdf import normpdf


def bsputcharm(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes European put charm = -dDelta_put/dT.

    Sign convention matches bscallcharm — delta bleed per year of
    calendar time passing.

    Per year. Divide by 252 for per-trading-day charm.

    Examples
    --------
    >>> bsputcharm(100.0, 100.0, 1.0, 0.05, 0.20)
    -0.11060933183045406
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    sigma_sqrt_T = c.sigma_safe * c.sqrt_T
    inner = (2.0 * (c.r - c.q) * c.T_safe - c.d2 * sigma_sqrt_T) / (2.0 * c.T_safe * sigma_sqrt_T)
    charm_analytic = (
        -c.q * xp.exp(-c.q * c.T_safe) * normcdf(-c.d1)
        - xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * inner
    )
    out = xp.where(c.normal, charm_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    out = xp.where(zero_vol, zero, out)

    # S=0 -> put_delta = -e^(-q·T), so d(delta)/dT = q·e^(-q·T),
    # and charm = -d(delta)/dT = -q·e^(-q·T).
    S_zero = (c.S <= 0) & (c.K > 0) & (c.T > 0)
    out = xp.where(S_zero, -c.q * xp.exp(-c.q * c.T), out)

    # K=0 -> put worthless -> delta=0 -> charm=0.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

'''Black-Scholes European call charm, batched.

charm_call = -∂Delta/∂T                      (delta decay per year of
                                              calendar time passing)

Closed form:
    charm_call = q·e^(-q·T)·Φ(d1)
                 - e^(-q·T)·φ(d1) · [ (2(r-q)T - d2·σ·√T) / (2·T·σ·√T) ]

Charm measures how quickly delta bleeds as time passes. Useful for
hedge-drift over holding periods, weekend gap risk on delta-neutral
portfolios, and pin-risk near expiry.

Design: docs/kernels/options/bscallcharm.md.
'''
from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf
from ..core.normpdf import normpdf


def bscallcharm(S, K, T, r, sigma, q=0.0):
    '''Black-Scholes European call charm = -dDelta/dT.

    Sign convention: charm = -∂Delta/∂T, i.e. the RATE OF DELTA BLEED
    as calendar time passes forward. A negative value for a long call
    means delta is INCREASING as time passes (unusual for OTM); the
    common ATM case has negative charm.

    Per year. Divide by 252 for per-trading-day charm.

    Examples
    --------
    >>> bscallcharm(100.0, 100.0, 1.0, 0.05, 0.20)
    -0.11060933183045406
    '''
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    sigma_sqrt_T = c.sigma_safe * c.sqrt_T
    inner = (2.0 * (c.r - c.q) * c.T_safe - c.d2 * sigma_sqrt_T) / (2.0 * c.T_safe * sigma_sqrt_T)
    charm_analytic = (
        c.q * xp.exp(-c.q * c.T_safe) * normcdf(c.d1)
        - xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * inner
    )
    out = xp.where(c.normal, charm_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Expired -> delta is a step function, charm undefined; conventionally 0.
    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    # Zero vol with T>0 -> delta jumps at forward-parity but is otherwise
    # constant, so charm is 0 away from the jump.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    out = xp.where(zero_vol, zero, out)

    # K=0 -> call = S·e^(-q·T), delta = e^(-q·T), charm = -q·e^(-q·T)·... = q·e^(-q·T)
    # d(delta)/dT = d(e^(-q·T))/dT = -q·e^(-q·T)
    K_zero = (c.K <= 0) & (c.S > 0)
    out = xp.where(K_zero, -c.q * xp.exp(-c.q * c.T), out)

    # S=0 -> delta=0 always -> charm=0.
    S_zero = c.S <= 0
    out = xp.where(S_zero, zero, out)

    return finalize(out)

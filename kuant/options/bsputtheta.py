"""Black-Scholes European put theta, batched.

theta_put = -[ S·e^(-q·T)·φ(d1)·σ / (2·√T) ]
            + r·K·e^(-r·T)·Φ(-d2)
            - q·S·e^(-q·T)·Φ(-d1)

Units: per unit time in years. Divide by 252 for per-trading-day theta,
or by 365 for per-calendar-day theta.

Sign: typically negative for long puts (option loses value with time),
but can be positive for deep ITM European puts with high interest rates.

Design: docs/kernels/options/bsputtheta.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf
from ..core.normpdf import normpdf


def bsputtheta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European put theta.

    Per year. For per-trading-day theta divide by 252; for per-
    calendar-day theta divide by 365.

    Examples
    --------
    >>> bsputtheta(100.0, 100.0, 1.0, 0.05, 0.20)
    -1.6579568392598896
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    term1 = -c.S_safe * xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * c.sigma_safe / (2.0 * c.sqrt_T)
    term2 = c.r * c.K_safe * xp.exp(-c.r * c.T_safe) * normcdf(-c.d2)
    term3 = -c.q * c.S_safe * xp.exp(-c.q * c.T_safe) * normcdf(-c.d1)
    theta_analytic = term1 + term2 + term3
    out = xp.where(c.normal, theta_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.K * xp.exp(-c.r * c.T) > c.S * xp.exp(-c.q * c.T)
    theta_zv_exercise = c.r * c.K * xp.exp(-c.r * c.T) - c.q * c.S * xp.exp(-c.q * c.T)
    theta_zero_vol = xp.where(always_exercise, theta_zv_exercise, zero)
    out = xp.where(zero_vol, theta_zero_vol, out)

    # S=0 with T>0 -> put worth K·e^(-r·T); theta = r·K·e^(-r·T).
    S_zero = (c.S <= 0) & (c.K > 0) & (c.T > 0)
    out = xp.where(S_zero, c.r * c.K * xp.exp(-c.r * c.T), out)

    # K=0 -> put worthless; theta 0.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

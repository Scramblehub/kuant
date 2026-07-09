"""Black-Scholes European call theta, batched.

theta_call = -[ S·e^(-q·T)·φ(d1)·σ / (2·√T) ]
             - r·K·e^(-r·T)·Φ(d2)
             + q·S·e^(-q·T)·Φ(d1)

Units: per unit time in years. Divide by 252 for "per trading day",
or by 365 for calendar-day theta.

Sign: typically negative for long calls (option loses value with
time), but can be positive for deep ITM European calls with high
dividends.

Design: docs/kernels/options/bscalltheta.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf
from ..core.normpdf import normpdf


def bscalltheta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European call theta.

    Per year. For per-trading-day theta divide by 252; for per-
    calendar-day theta divide by 365.

    Examples
    --------
    >>> bscalltheta(100.0, 100.0, 1.0, 0.05, 0.20)
    -6.414027546438197
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    term1 = -c.S_safe * xp.exp(-c.q * c.T_safe) * normpdf(c.d1) * c.sigma_safe / (2.0 * c.sqrt_T)
    term2 = -c.r * c.K_safe * xp.exp(-c.r * c.T_safe) * normcdf(c.d2)
    term3 = c.q * c.S_safe * xp.exp(-c.q * c.T_safe) * normcdf(c.d1)
    theta_analytic = term1 + term2 + term3
    out = xp.where(c.normal, theta_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)

    # Expired -> theta undefined but conventionally 0 (no more time value).
    expired = (c.T <= 0) & (c.S >= 0) & (c.K >= 0)
    out = xp.where(expired, zero, out)

    # Zero vol, T>0 -> theta is deterministic; only the discount-rate
    # term remains. If the call is guaranteed to exercise, theta reflects
    # the growth of the exercise cost.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.S * xp.exp(-c.q * c.T) > c.K * xp.exp(-c.r * c.T)
    theta_zv_exercise = -c.r * c.K * xp.exp(-c.r * c.T) + c.q * c.S * xp.exp(-c.q * c.T)
    theta_zero_vol = xp.where(always_exercise, theta_zv_exercise, zero)
    out = xp.where(zero_vol, theta_zero_vol, out)

    # K=0 -> call equals discounted S; theta = q·S·e^(-q·T).
    K_zero = (c.K <= 0) & (c.S > 0)
    out = xp.where(K_zero, c.q * c.S * xp.exp(-c.q * c.T), out)

    # S=0 -> worthless option; theta 0.
    S_zero = c.S <= 0
    out = xp.where(S_zero, zero, out)

    return finalize(out)

"""Black-Scholes power option price, batched.

A power option's terminal payoff is a power of the underlying minus a
strike:

    power_call = max(S_T^n - K, 0)
    power_put  = max(K - S_T^n, 0)

For n > 1 the payoff convex-amplifies gains beyond the strike; for
n in (0, 1) it diminishes them. n = 1 recovers plain vanilla.

Closed form under Black-Scholes with continuous dividend q:

    Let mu_n = n*(r - q) + n*(n - 1)*sigma^2/2
    d1_n = (log(S/K^(1/n)) + (r - q + (n - 1/2)*sigma^2)*T) / (sigma*sqrt(T))
    d2_n = d1_n - n*sigma*sqrt(T)
    power_call = S^n * exp((mu_n - r)*T) * N(d1_n) - K*exp(-r*T)*N(d2_n)

with the put obtained by the standard sign flips. The `exp((mu_n - r)*T)`
factor accounts for the risk-neutral drift of `S_t^n` under the standard
Black-Scholes measure.

References
----------
Heynen & Kat 1996, "Pricing and hedging power options." Haug 2007,
"The Complete Guide to Option Pricing Formulas", chapter 4.24.

Design: docs/kernels/options/powerprice.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf


def powerprice(S, K, T, r, sigma, q=0.0, *, n=2.0, is_call=True):
    """Black-Scholes power option pricing.

    Parameters
    ----------
    S, K, T, r, sigma, q : scalar or array
        Standard Black-Scholes inputs. `K` is the strike compared
        against `S_T^n`.
    n : scalar or array, default 2.0
        Power exponent. `n = 1` recovers plain vanilla.
    is_call : bool, default True

    Returns
    -------
    scalar or array

    Examples
    --------
    Squared call, ATM (K = S^2 = 10000):

    >>> round(powerprice(100.0, 10000.0, 1.0, 0.05, 0.20, n=2.0), 2)
    2432.7
    """
    # Scalar 'n' guard: KE-VAL-RANGE. Array-typed 'n' is not batched here
    # (power option payoffs are typically evaluated at a single exponent),
    # so a scalar float(n) is the expected input.
    try:
        n_scalar = float(n)
    except (TypeError, ValueError):
        n_scalar = None
    if n_scalar is not None and n_scalar == 0.0:
        raise ValueError(
            "kuant.powerprice: 'n' must be non-zero (payoff S^n is "
            "undefined/degenerate at n=0).  [KE-VAL-RANGE]"
        )

    # prepare_bs is called with a dummy strike; we compute d1_n, d2_n fresh.
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    n_arr = xp.asarray(n, dtype=c.out_dtype)
    sqrt_T = c.sqrt_T
    sigma_safe = c.sigma_safe

    # K^(1/n) is the effective spot-equivalent strike for the n-th root
    # substitution S^n vs K.
    K_root_n = xp.where(c.K > 0, c.K_safe ** (1 / n_arr), c.K_safe)
    log_ratio = xp.log(c.S_safe / K_root_n)
    d1_n = (log_ratio + (c.r - c.q + (n_arr - 0.5) * sigma_safe * sigma_safe) * c.T_safe) / (
        sigma_safe * sqrt_T
    )
    d2_n = d1_n - n_arr * sigma_safe * sqrt_T

    mu_n = n_arr * (c.r - c.q) + n_arr * (n_arr - 1) * sigma_safe * sigma_safe / 2
    disc_r = xp.exp(-c.r * c.T_safe)
    drift_adj = xp.exp((mu_n - c.r) * c.T_safe)

    if is_call:
        price = c.S_safe**n_arr * drift_adj * normcdf(d1_n) - c.K_safe * disc_r * normcdf(d2_n)
    else:
        price = c.K_safe * disc_r * normcdf(-d2_n) - c.S_safe**n_arr * drift_adj * normcdf(-d1_n)

    c.out = xp.where(
        (c.T > 0) & (c.sigma > 0) & (c.S > 0) & (c.K > 0),
        price,
        c.out,
    )
    return finalize(c.out)


__all__ = ["powerprice"]

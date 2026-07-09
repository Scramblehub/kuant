"""Black-Scholes floating-strike lookback option price, batched.

Floating-strike lookback pays the difference between the terminal
price and the extremum reached over the option's life:

    lookback_call = S_T - min_{0 <= t <= T} S_t     (call: buy at the low)
    lookback_put  = max_{0 <= t <= T} S_t - S_T     (put:  sell at the high)

Closed form under continuous monitoring (Goldman-Sosin-Gatto 1979):

    b = r - q
    a1 = (log(S/S_min) + (b + sigma^2/2)*T) / (sigma*sqrt(T))
    a2 = a1 - sigma*sqrt(T)
    lookback_call = S*exp(-q*T)*N(a1)
                  - S_min*exp(-r*T)*N(a2)
                  - S*exp(-q*T)*(sigma^2/(2*b))*(N(-a1)
                     - exp(b*T)*(S_min/S)^(2*b/sigma^2)*N(-a1 + 2*b*sqrt(T)/sigma))

with a symmetric formula for the put. When `b = 0` a special limiting
form is used; this kernel implements the general case with a small-b
guard.

For a fresh contract (no history), pass `S_extreme = S`.

References
----------
Goldman, Sosin & Gatto 1979, "Path Dependent Options: Buy at the Low,
Sell at the High." Haug 2007, chapter 5.13.

Design: docs/kernels/options/lookbackprice.md.
"""

from __future__ import annotations

import warnings

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf
from ..errors import KuantNumericWarning


def lookbackprice(S, S_extreme, T, r, sigma, q=0.0, *, is_call=True):
    """Black-Scholes floating-strike lookback under continuous monitoring.

    Parameters
    ----------
    S : scalar or array
        Current underlying price.
    S_extreme : scalar or array
        Running minimum (for calls) or maximum (for puts) observed so
        far. Pass `S` for a fresh contract with no history.
    T, r, sigma, q : scalar or array
        Standard Black-Scholes inputs.
    is_call : bool, default True

    Returns
    -------
    scalar or array

    Examples
    --------
    Fresh ATM contract, continuous monitoring:

    >>> round(lookbackprice(100.0, 100.0, 1.0, 0.05, 0.20), 4)
    17.2168
    """
    # `prepare_bs` computes standard d1, d2 with strike = S_extreme,
    # which we DON'T want for the lookback. We pass K = S_extreme just to
    # get a sanely shaped context, then compute a1, a2 fresh below.
    c = prepare_bs(S, S_extreme, T, r, sigma, q)
    xp = c.xp

    b = c.r - c.q
    # Guard b -> 0 with a small epsilon to avoid 1/b explosion.
    # For scalar inputs where |b| falls into the guarded region, emit a
    # KW-* soft warning: the sigma^2/(2b) prefactor amplifies numerical
    # error there and the closed form should ideally be replaced with the
    # b=0 limiting series expansion. Users should treat the price as
    # indicative in that regime.
    try:
        b_scalar = float(b)
        if abs(b_scalar) < 1e-6:
            warnings.warn(
                f"kuant.lookbackprice: |b|=|r-q|={abs(b_scalar):.2e} is "
                f"near zero; the sigma^2/(2b) prefactor is guarded but "
                f"the returned price loses precision. Prefer the b=0 "
                f"limiting form for production use.  "
                f"[KW-LOOKBACK-NEAR-ZERO-CARRY]",
                KuantNumericWarning,
                stacklevel=2,
            )
    except (TypeError, ValueError):
        pass
    b_safe = xp.where(xp.abs(b) < 1e-8, xp.asarray(1e-8, dtype=c.out_dtype), b)

    sqrt_T = c.sqrt_T
    sigma_safe = c.sigma_safe
    S_ext = xp.asarray(S_extreme, dtype=c.out_dtype)

    log_ratio = xp.log(c.S_safe / xp.where(S_ext > 0, S_ext, c.S_safe))
    a1 = (log_ratio + (b_safe + sigma_safe * sigma_safe / 2) * c.T_safe) / (sigma_safe * sqrt_T)
    a2 = a1 - sigma_safe * sqrt_T

    disc_q = xp.exp(-c.q * c.T_safe)
    disc_r = xp.exp(-c.r * c.T_safe)
    two_b_over_sig2 = 2 * b_safe / (sigma_safe * sigma_safe)
    ext_ratio_pow = (S_ext / c.S_safe) ** two_b_over_sig2

    # Haug 2007 chapter 5.13 formulation.
    kicker = 2 * b_safe * sqrt_T / sigma_safe

    if is_call:
        # S_extreme is the running minimum (m).
        adj = (sigma_safe * sigma_safe / (2 * b_safe)) * (
            ext_ratio_pow * normcdf(-a1 + kicker) - xp.exp(b_safe * c.T_safe) * normcdf(-a1)
        )
        price = (
            c.S_safe * disc_q * normcdf(a1) - S_ext * disc_r * normcdf(a2) + c.S_safe * disc_r * adj
        )
    else:
        # S_extreme is the running maximum (M).
        adj = (sigma_safe * sigma_safe / (2 * b_safe)) * (
            -ext_ratio_pow * normcdf(a1 - kicker) + xp.exp(b_safe * c.T_safe) * normcdf(a1)
        )
        price = (
            S_ext * disc_r * normcdf(-a2)
            - c.S_safe * disc_q * normcdf(-a1)
            + c.S_safe * disc_r * adj
        )

    c.out = xp.where(
        (c.T > 0) & (c.sigma > 0) & (c.S > 0) & (S_ext > 0),
        price,
        c.out,
    )
    return finalize(c.out)


__all__ = ["lookbackprice"]

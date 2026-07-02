"""Black-Scholes European put delta, batched.

delta = -exp(-q*T) * Phi(-d1)

Range: [-1, 0]. Directly powers the M9 TP monitor's D70 rule.

Refactored to use kuant.core._bs_common.prepare_bs.

EDGE CASES (delta-specific — not all collapse to zero):
  - T <= 0 (expired): -1 if K > S else 0 (step function at kink)
  - sigma <= 0: same step, but at forward-adjusted break-even
  - S == 0: -exp(-q*T) (guaranteed exercise)
  - K == 0: 0
"""
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bsputdelta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European put delta.

    Returns values in [-1, 0].

    Notes
    -----
    delta = -exp(-q*T) * Phi(-d1)

    Examples
    --------
    >>> bsputdelta(100.0, 100.0, 1.0, 0.05, 0.2)  # ATM
    -0.3631693488243809
    >>> bsputdelta(100.0, 120.0, 1.0, 0.05, 0.2)  # deep ITM
    -0.7128083620948729

    See Also
    --------
    kuant.core.bsput : bsputdelta is its dS-derivative.
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    # One line of actual math
    delta_analytic = -xp.exp(-c.q * c.T_safe) * normcdf(-c.d1)
    out = xp.where(c.normal, delta_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    neg_one_disc = -xp.exp(-c.q * c.T)  # -exp(-q*T), "guaranteed exercise" delta

    # Case A: expired -> step at K.
    expired = (c.T <= 0) & (c.S > 0) & (c.K > 0)
    delta_expired = xp.where(c.K > c.S, xp.asarray(-1.0, dtype=c.out_dtype), zero)
    out = xp.where(expired, delta_expired, out)

    # Case B: zero vol, T > 0 -> deterministic step at forward break-even.
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.K * xp.exp(-c.r * c.T) > c.S * xp.exp(-c.q * c.T)
    delta_zero_vol = xp.where(always_exercise, neg_one_disc, zero)
    out = xp.where(zero_vol, delta_zero_vol, out)

    # Case C: S == 0 -> guaranteed exercise.
    S_zero = (c.S <= 0) & (c.K > 0)
    out = xp.where(S_zero, neg_one_disc, out)

    # Case D: K == 0 -> put never exercises.
    K_zero = c.K <= 0
    out = xp.where(K_zero, zero, out)

    return finalize(out)

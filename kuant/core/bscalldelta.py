"""Black-Scholes European call delta, batched.

delta = +exp(-q*T) * Phi(d1)

Range: [0, 1]. Note the sign and Phi-argument flip vs bsputdelta:
    put:  -exp(-q*T) * Phi(-d1)   -> [-1, 0]
    call: +exp(-q*T) * Phi(d1)    -> [0, 1]

Related by: delta_call - delta_put = exp(-q*T)  (put-call parity for delta).

Uses kuant.core._bs_common.prepare_bs.

EDGE CASES (mirror bsputdelta with swapped S=0/K=0 handling):
  - T <= 0 (expired): +1 if S > K else 0 (step function)
  - sigma <= 0, T > 0: +exp(-q*T) if forward > K, else 0
  - S == 0: 0 (call worthless — nothing to buy)
  - K == 0: +exp(-q*T) (guaranteed exercise)
"""
from __future__ import annotations

from ._bs_common import finalize, prepare_bs
from .normcdf import normcdf


def bscalldelta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes European call delta. Range [0, 1].

    Notes
    -----
    delta = exp(-q*T) * Phi(d1)

    Examples
    --------
    >>> bscalldelta(100.0, 100.0, 1.0, 0.05, 0.20)
    0.6368306511756191
    >>> bscalldelta(100.0, 120.0, 1.0, 0.05, 0.20)  # OTM
    0.28719163790512714

    See Also
    --------
    kuant.core.bscall : bscalldelta is its dS-derivative.
    kuant.core.bsputdelta : Related by delta_call - delta_put = exp(-q*T).
    """
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    # One line of actual math
    delta_analytic = xp.exp(-c.q * c.T_safe) * normcdf(c.d1)
    out = xp.where(c.normal, delta_analytic, c.out)

    zero = xp.asarray(0.0, dtype=c.out_dtype)
    one_disc = xp.exp(-c.q * c.T)  # +exp(-q*T), "guaranteed exercise" delta

    # Case A: expired -> step at K.
    expired = (c.T <= 0) & (c.S > 0) & (c.K > 0)
    delta_expired = xp.where(c.S > c.K, xp.asarray(1.0, dtype=c.out_dtype), zero)
    out = xp.where(expired, delta_expired, out)

    # Case B: zero vol, T > 0 -> deterministic step at forward break-even.
    # Call exercises if S*exp((r-q)*T) > K, i.e. S*exp(-q*T) > K*exp(-r*T).
    zero_vol = (c.sigma <= 0) & (c.T > 0) & (c.S > 0) & (c.K > 0)
    always_exercise = c.S * xp.exp(-c.q * c.T) > c.K * xp.exp(-c.r * c.T)
    delta_zero_vol = xp.where(always_exercise, one_disc, zero)
    out = xp.where(zero_vol, delta_zero_vol, out)

    # Case C: K == 0 -> guaranteed exercise -> +exp(-q*T).
    # (Put has this at S=0; swapped for call.)
    K_zero = (c.K <= 0) & (c.S > 0)
    out = xp.where(K_zero, one_disc, out)

    # Case D: S == 0 -> call worthless -> 0.
    # (Put has this at K=0; swapped for call.)
    S_zero = c.S <= 0
    out = xp.where(S_zero, zero, out)

    return finalize(out)

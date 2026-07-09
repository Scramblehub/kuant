"""Black-Scholes simple chooser option price, batched.

A simple chooser gives the holder the right at chooser date `t_choose`
to declare whether the option is a call or a put. Both hypothetical
options have the same strike `K` and remaining maturity `T - t_choose`.

At `t_choose` the holder picks max(C, P) where C, P are the values of
the two hypothetical vanillas evaluated with time to expiry (T - t_choose).

Rubinstein 1991 closed form for the simple chooser at t=0:

    chooser = call(S, K, T; sigma) + put(S, K * exp(-(r-q)*(T-t_choose)), t_choose; sigma)

The equivalence between this and the "max(call, put) at chooser time"
formulation is proved via put-call parity.

Design: docs/kernels/options/chooserprice.md.
"""

from __future__ import annotations

from ..core._bs_common import finalize, prepare_bs
from ..core.normcdf import normcdf


def chooserprice(S, K, T, t_choose, r, sigma, q=0.0):
    """Black-Scholes simple chooser (Rubinstein 1991).

    Parameters
    ----------
    S, K, T, r, sigma, q : scalar or array
        Standard Black-Scholes inputs. T is total time to expiry.
    t_choose : scalar or array
        Time at which the holder must choose call vs put. Must satisfy
        `0 <= t_choose <= T`.

    Returns
    -------
    scalar or array

    Examples
    --------
    Chooser at half-life, ATM:

    >>> round(chooserprice(100.0, 100.0, 1.0, 0.5, 0.05, 0.20), 4)
    13.8513
    """
    # Scalar guard on t_choose against T. Array-broadcast t_choose fields
    # (rare in practice) are still handled at the `where` mask below.
    try:
        t_choose_val = float(t_choose)
        T_val = float(T)
        if t_choose_val < 0.0 or t_choose_val > T_val:
            raise ValueError(
                f"kuant.chooserprice: 't_choose' ({t_choose_val}) must "
                f"satisfy 0 <= t_choose <= T ({T_val}).  [KE-VAL-RANGE]"
            )
    except (TypeError, ValueError) as exc:
        if "KE-VAL-RANGE" in str(exc):
            raise

    # First term: standard vanilla call with full time T.
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp

    disc_r_T = xp.exp(-c.r * c.T_safe)
    disc_q_T = xp.exp(-c.q * c.T_safe)
    call_full = c.S_safe * disc_q_T * normcdf(c.d1) - c.K_safe * disc_r_T * normcdf(c.d2)

    # Second term: put with strike K * exp(-(r-q)*(T - t_choose)) and
    # time-to-expiry t_choose. Use a second prepare_bs call.
    t_choose_arr = xp.asarray(t_choose, dtype=c.out_dtype)
    K_shifted = c.K_safe * xp.exp(-(c.r - c.q) * (c.T_safe - t_choose_arr))
    c2 = prepare_bs(S, K_shifted, t_choose_arr, c.r, c.sigma, c.q)
    disc_r_tc = xp.exp(-c2.r * c2.T_safe)
    disc_q_tc = xp.exp(-c2.q * c2.T_safe)
    put_tc = K_shifted * disc_r_tc * normcdf(-c2.d2) - c2.S_safe * disc_q_tc * normcdf(-c2.d1)

    price = call_full + put_tc
    c.out = xp.where(
        (c.T > 0)
        & (c.sigma > 0)
        & (c.S > 0)
        & (c.K > 0)
        & (t_choose_arr >= 0)
        & (t_choose_arr <= c.T),
        price,
        c.out,
    )
    return finalize(c.out)


__all__ = ["chooserprice"]

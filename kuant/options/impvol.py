"""Implied volatility solver via vectorized Newton-Raphson.

Given market prices and (S, K, T, r, q), find the sigma that reproduces
each price under Black-Scholes:

    bsput(S, K, T, r, sigma_iv, q) == price   (or bscall for calls)

Vectorized: all elements iterate simultaneously; converged elements are
frozen. Loop terminates when every element is within `tol` of its target
or `max_iter` is reached.

Initial guess: Manaster-Koehler
    sigma_0 = sqrt(|ln(S/K) + r*T| * 2 / T)

Robust across moneyness. Newton typically converges in 3-5 iterations for
in-bounds prices.

Failure modes → NaN:
  - Price outside no-arbitrage bounds
  - Vega too small (curve is flat; solver useless)
  - Max iterations without converging to `tol * 10`

Design: docs/kernels/impvol.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_positive, warn_kuant
from kuant.errors import KuantNumericWarning

from ..core import bscall, bsput
from .bsvega import bsvega

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


# Solver knobs (module-level constants — override at call site if needed)
_SIGMA_MIN = 1e-6  # smallest sigma we'll consider (essentially zero vol)
_SIGMA_MAX = 10.0  # largest sigma we'll consider (1000% annualized)
_VEGA_MIN = 1e-8  # below this, Newton step is unusable
_FINAL_TOL_MULT = 10  # widen tol by this factor for final validation


def _detect_backend(*args) -> Any:
    for a in args:
        if isinstance(a, _CUPY_NDARRAY):
            return cp
    return np


def impvol(price, S, K, T, r, is_call=False, q=0.0, tol=1e-8, max_iter=100):
    """Vectorized implied-vol solver.

    Parameters
    ----------
    price : scalar or array
        Observed option price(s). Must be in the no-arbitrage range for
        the corresponding (S, K, T, r, q).
    S, K, T, r : scalar or array
        Spot, strike, tenor (years), risk-free rate (decimal).
    is_call : bool, default False
        True for call, False for put. Broadcasts to all elements.
    q : scalar or array, default 0.0
        Continuous dividend yield (decimal).
    tol : float, default 1e-8
        Convergence tolerance on the price residual.
    max_iter : int, default 100
        Maximum Newton iterations. Elements not converged after this get NaN.

    Returns
    -------
    sigma : scalar or array
        Implied volatility. Shape follows the broadcast of the inputs.
        NaN where the input is out-of-arbitrage-bounds or the solver
        failed to converge.

    Examples
    --------
    >>> from kuant.core import bsput
    >>> sigma_true = 0.30
    >>> price = bsput(100.0, 105.0, 0.5, 0.05, sigma_true)
    >>> sigma_iv = impvol(price, 100.0, 105.0, 0.5, 0.05)
    >>> abs(sigma_iv - sigma_true) < 1e-8
    True
    """
    require_positive(max_iter, "max_iter", kernel="impvol", kind="int")
    require_positive(tol, "tol", kernel="impvol")

    xp = _detect_backend(price, S, K, T, r, q)

    # Coerce and broadcast all inputs.
    price = xp.asarray(price)
    S = xp.asarray(S)
    K = xp.asarray(K)
    T = xp.asarray(T)
    r = xp.asarray(r)
    q = xp.asarray(q)

    # Pick dtype from required args; q's default 0.0 shouldn't force float64.
    required_dtypes = [price.dtype, S.dtype, K.dtype, T.dtype, r.dtype]
    out_dtype = xp.result_type(*required_dtypes)
    if out_dtype.kind in "iub":
        out_dtype = xp.dtype(xp.float64)

    # Cast every input to out_dtype, then broadcast to common shape.
    price = price.astype(out_dtype, copy=False)
    S = S.astype(out_dtype, copy=False)
    K = K.astype(out_dtype, copy=False)
    T = T.astype(out_dtype, copy=False)
    r = r.astype(out_dtype, copy=False)
    q = q.astype(out_dtype, copy=False)
    price, S, K, T, r, q = xp.broadcast_arrays(price, S, K, T, r, q)

    was_scalar = price.ndim == 0

    # No-arbitrage bounds.
    if is_call:
        lower = xp.maximum(
            S * xp.exp(-q * T) - K * xp.exp(-r * T), xp.asarray(0.0, dtype=out_dtype)
        )
        upper = S * xp.exp(-q * T)
    else:
        lower = xp.maximum(
            K * xp.exp(-r * T) - S * xp.exp(-q * T), xp.asarray(0.0, dtype=out_dtype)
        )
        upper = K * xp.exp(-r * T)

    in_bounds = (price >= lower) & (price <= upper) & (T > 0) & (S > 0) & (K > 0)

    # Manaster-Koehler initial guess: sigma_0 = sqrt(|ln(S/K) + r*T| * 2 / T)
    T_safe = xp.where(T > 0, T, xp.asarray(1.0, dtype=out_dtype))
    S_safe = xp.where(S > 0, S, xp.asarray(1.0, dtype=out_dtype))
    K_safe = xp.where(K > 0, K, xp.asarray(1.0, dtype=out_dtype))
    sigma = xp.sqrt(xp.abs(xp.log(S_safe / K_safe) + r * T_safe) * 2.0 / T_safe)
    sigma = xp.clip(sigma, _SIGMA_MIN, _SIGMA_MAX)

    # Newton loop.
    for _ in range(max_iter):
        pv = bscall(S, K, T, r, sigma, q) if is_call else bsput(S, K, T, r, sigma, q)
        residual = pv - price

        # Element-wise convergence and vega checks.
        converged = xp.abs(residual) < tol
        # Early break if every element (in-bounds subset) is converged.
        if bool(xp.all(converged | ~in_bounds)):
            break

        vega = bsvega(S, K, T, r, sigma, q)
        vega_ok = vega > _VEGA_MIN
        vega_safe = xp.where(vega_ok, vega, xp.asarray(1.0, dtype=out_dtype))
        step = residual / vega_safe

        sigma_next = xp.clip(sigma - step, _SIGMA_MIN, _SIGMA_MAX)

        # Only advance elements that (a) haven't converged and (b) had usable vega.
        should_update = (~converged) & vega_ok & in_bounds
        sigma = xp.where(should_update, sigma_next, sigma)

    # Final validation: residual within FINAL_TOL_MULT * tol AND input was in bounds.
    pv_final = bscall(S, K, T, r, sigma, q) if is_call else bsput(S, K, T, r, sigma, q)
    residual_final = xp.abs(pv_final - price)
    ok = in_bounds & (residual_final < tol * _FINAL_TOL_MULT)

    # Post-loop diagnostics for cells that did NOT converge but were in
    # bounds. Two independent failure modes surface here.
    unconverged_in_bounds = in_bounds & ~ok
    n_unconv = (
        int(unconverged_in_bounds.sum()) if xp is np else int(unconverged_in_bounds.sum().get())
    )
    if n_unconv > 0:
        warn_kuant(
            kernel="impvol",
            code="KW-CONV-MAX-ITER",
            what=(
                f"{n_unconv} in-bounds cells failed to converge within "
                f"max_iter={int(max_iter)} at tol={tol:g}; those cells "
                f"return NaN"
            ),
            fix=(
                "increase max_iter, loosen tol, or fall back to "
                "impvolbisection for the failing cells"
            ),
            category=KuantNumericWarning,
        )
    # Detect vega-flat cells: recompute vega at the returned sigma and
    # count where it fell below the Newton floor. This can co-occur with
    # non-convergence above but is a distinct diagnosis.
    vega_final = bsvega(S, K, T, r, sigma, q)
    vega_degen = in_bounds & (vega_final < _VEGA_MIN)
    n_vega = int(vega_degen.sum()) if xp is np else int(vega_degen.sum().get())
    if n_vega > 0:
        warn_kuant(
            kernel="impvol",
            code="KW-NUM-VEGA-DEGENERATE",
            what=(
                f"vega fell below {_VEGA_MIN:g} in {n_vega} cells; Newton "
                f"step was unusable in those cells (likely deep-OTM or "
                f"very short tenor)"
            ),
            fix=(
                "use impvolbisection for deep-OTM / very-short-tenor cells "
                "where the price-to-vol map is nearly flat"
            ),
            category=KuantNumericWarning,
        )

    sigma = xp.where(ok, sigma, xp.asarray(xp.nan, dtype=out_dtype))

    if was_scalar:
        return float(sigma)
    return sigma

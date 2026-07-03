"""Exponentially weighted standard deviation via recurrence.

Follows pandas ewm(..., adjust=False).std() semantics.

Uses two coupled EMA recurrences:
    m1[i] = alpha * x[i]      + (1-alpha) * m1[i-1]    (mean)
    m2[i] = alpha * x[i]²     + (1-alpha) * m2[i-1]    (second moment)

Then var[i] = m2[i] - m1[i]²  and std[i] = sqrt(var[i]).

Bias correction (pandas default: bias=False):
    var_debias = var / (1 - w_sq_sum / w_sum²)
where w_sum and w_sq_sum are running sums of exponential weights.

Design: docs/kernels/rollemastd.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _prepare_input(x):
    if isinstance(x, _CUPY_NDARRAY):
        arr = x
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        backend = cp
    else:
        arr = np.asarray(x)
        if arr.dtype.kind in "iub":
            arr = arr.astype(np.float64)
        backend = np

    require_1d(arr, "x", kernel="rollemastd")

    return backend, arr, arr.dtype


def _ema_via_lfilter(arr_np, alpha):
    """Run the recurrence y[n] = alpha*x[n] + (1-alpha)*y[n-1] with
    initial condition making y[0] = x[0]."""
    b = np.asarray([alpha])
    a = np.asarray([1.0, -(1.0 - alpha)])
    zi = np.asarray([(1.0 - alpha) * arr_np[0]])
    out, _ = lfilter(b, a, arr_np, zi=zi)
    return out


def rollemastd(x, span=None, alpha=None, bias=False):
    """Exponentially weighted standard deviation.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
    span : float, optional
        pandas convention: alpha = 2 / (span + 1). Must be >= 1.
    alpha : float, optional
        Smoothing factor in (0, 1].
    bias : bool, default False
        If True, return the biased estimator (no debias correction).
        Default False matches pandas ewm().std(bias=False).

    Returns
    -------
    1D array, same length/backend/dtype as x.

    Notes
    -----
    Matches pandas ewm(..., adjust=False, bias=bias).std() to machine
    precision on the sample cases. The first entry is NaN because
    a single sample has no dispersion.
    """
    if (span is None) == (alpha is None):
        both = span is not None
        raise KuantValueError(
            "kuant.rollemastd: provide exactly one of `span` or `alpha`, got "
            f"{'both' if both else 'neither'}.  [KE-VAL-MUTEX]\n"
            "  → Fix: `span=21` (pandas-style, alpha = 2/(span+1)) OR "
            "`alpha=0.1` (direct smoothing factor)"
        )
    if span is not None:
        require_range(span, "span", kernel="rollemastd", lo=1.0, hi=float("inf"))
        alpha_val = 2.0 / (float(span) + 1.0)
    else:
        assert alpha is not None
        alpha_val = float(alpha)
        require_range(
            alpha_val,
            "alpha",
            kernel="rollemastd",
            lo=0.0,
            hi=1.0,
            lo_inclusive=False,
            hi_inclusive=True,
        )

    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    if n == 0:
        return arr

    arr_np = arr if xp is np else cp.asnumpy(arr)

    beta = 1.0 - alpha_val

    # Recurrences for m1 and m2.
    m1 = _ema_via_lfilter(arr_np, alpha_val)
    m2 = _ema_via_lfilter(arr_np * arr_np, alpha_val)

    var_biased = m2 - m1 * m1
    var_biased = np.maximum(var_biased, 0.0)  # guard FP negatives

    if bias:
        var_out = var_biased
    else:
        # For adjust=False, at step k (n_obs = k) the weights on the k
        # observations are (from newest to oldest):
        #   α, αβ, αβ², ..., αβ^(k-2), β^(k-1)
        # Sum = 1 (verified algebraically). Sum-of-squares:
        #   Σw² = α² · (1 - β^(2(k-1))) / (1 - β²) + β^(2(k-1))
        n_obs = np.arange(1, n + 1, dtype=np.float64)
        if beta < 1.0:
            beta_2km2 = beta ** (2.0 * (n_obs - 1.0))
            sum_w_sq = (alpha_val**2) * (1.0 - beta_2km2) / (1.0 - beta**2) + beta_2km2
        else:  # α == 1 → only newest observation weighted → Σw² = 1 always
            sum_w_sq = np.ones(n, dtype=np.float64)
        # Debias: since Σw = 1, unbiased_var = biased_var / (1 - Σw²).
        # At k=1, Σw²=1, so 1-Σw²=0 → NaN (undefined variance of single sample).
        one_minus_ratio = 1.0 - sum_w_sq
        with np.errstate(divide="ignore", invalid="ignore"):
            var_out = np.where(one_minus_ratio > 0, var_biased / one_minus_ratio, np.nan)

    std_out = np.sqrt(var_out).astype(out_dtype, copy=False)

    if xp is np:
        return std_out
    return xp.asarray(std_out)

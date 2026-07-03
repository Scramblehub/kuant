"""Exponentially weighted moving average via recurrence.

    ema[0] = x[0]
    ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]

Different algorithmic pattern from the window-based primitives — no
sliding window, no cumsum trick. The recurrence has an inherent
sequential dependency; the CPU path uses `scipy.signal.lfilter` (an
IIR filter implemented in compiled C) so the loop stays fast.

GPU path: for V1 we transfer to CPU, compute, and transfer back.
A native GPU implementation would need a parallel prefix scan on the
recurrence, which we haven't written yet. Documented as future work.

Parameters:
    span  — pandas convention: alpha = 2 / (span + 1)
    alpha — smoothing factor directly; must be in (0, 1]
Exactly one of `span` / `alpha` must be provided.

NaN policy: NaN propagates through the recurrence naturally. No shift
trick needed (recurrence is scale-invariant in the linear regime).

Design: docs/kernels/rollema.md.
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

    require_1d(arr, "x", kernel="rollema")

    return backend, arr, arr.dtype


def rollema(x, span=None, alpha=None):
    """Exponentially weighted moving average.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
        Input values.
    span : float, optional
        pandas-style span. alpha = 2 / (span + 1). Must be >= 1.
    alpha : float, optional
        Smoothing factor in (0, 1]. Higher alpha = faster response.

    Exactly one of `span` or `alpha` must be provided.

    Returns
    -------
    1D array, same length/backend/dtype as x.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2, 3, 4, 5])
    >>> rollema(x, alpha=0.5)
    array([1.    , 1.5   , 2.25  , 3.125 , 4.0625])
    """
    if (span is None) == (alpha is None):
        both = span is not None
        raise KuantValueError(
            "kuant.rollema: provide exactly one of `span` or `alpha`, got "
            f"{'both' if both else 'neither'}.  [KE-VAL-MUTEX]\n"
            "  → Fix: `span=21` (pandas-style, alpha = 2/(span+1)) OR "
            "`alpha=0.1` (direct smoothing factor)"
        )

    if span is not None:
        require_range(span, "span", kernel="rollema", lo=1.0, hi=float("inf"))
        alpha_val = 2.0 / (float(span) + 1.0)
    else:
        # Guaranteed non-None by the exactly-one check above.
        assert alpha is not None
        alpha_val = float(alpha)
        require_range(
            alpha_val,
            "alpha",
            kernel="rollema",
            lo=0.0,
            hi=1.0,
            lo_inclusive=False,
            hi_inclusive=True,
        )

    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size

    if n == 0:
        return arr

    # scipy.signal.lfilter needs numpy — transfer if GPU.
    # Access cp explicitly (typed Any) so Pylance doesn't narrow xp
    # to numpy in the else branch.
    arr_np = arr if xp is np else cp.asnumpy(arr)

    # IIR form of the recurrence:
    #   y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    #   -->  y[n] - (1-alpha)*y[n-1] = alpha*x[n]
    #   -->  b = [alpha], a = [1, -(1-alpha)]
    #
    # Initial condition zi chosen so y[0] = x[0]:
    #   y[0] = b[0]*x[0] + zi[0]
    #        = alpha*x[0] + zi[0] = x[0]
    #   -->  zi[0] = (1 - alpha) * x[0]
    b = np.asarray([alpha_val])
    a = np.asarray([1.0, -(1.0 - alpha_val)])
    zi = np.asarray([(1.0 - alpha_val) * arr_np[0]])

    result_np, _ = lfilter(b, a, arr_np, zi=zi)
    result_np = result_np.astype(out_dtype, copy=False)

    if xp is np:
        return result_np
    return xp.asarray(result_np)

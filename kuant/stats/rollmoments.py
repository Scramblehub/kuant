"""Rolling higher-moment kernels: rollskew (3rd) and rollkurt (4th, excess).

Both use the cumsum trick on x, x², x³ (and x⁴ for kurt), with the shifting
stability trick borrowed from rollstd. Bias corrections match pandas
`.rolling().skew()` and `.rolling().kurt()` (Fisher / adjusted-Fisher-Pearson).

Math for a window of size w with shifted values y = x - x[0]:
    m2 = (S2 - S1²/w) / w                # 2nd central moment
    m3 = (S3 - 3·μ·S2 + 2·w·μ³) / w      # 3rd central moment, μ = S1/w
    m4 = (S4 - 4·μ·S3 + 6·μ²·S2 - 3·w·μ⁴) / w   # 4th central moment

Both central moments are shift-invariant (y ↦ x). So the answer is exact,
but small y-values keep cancellation minimal.

Bias-corrected outputs:
    skew  = √(w(w-1)) / (w-2)          · m3 / m2^(3/2)          (requires w >= 3)
    kurt  = (w-1) / ((w-2)(w-3)) · ((w+1)·m4/m2² - 3(w-1))      (requires w >= 4)
          (excess kurtosis, matches pandas)

NaN policy — STRICT WINDOW. Zero-variance guard: m2 == 0 → NaN.

Design: docs/kernels/rollmoments.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d, require_positive

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

    require_1d(arr, "x", kernel="rollmoments")

    return backend, arr, arr.dtype


def _rolling_moments_setup(x, window, up_to_order):
    """Compute shared cumsum-based intermediates for higher moments.

    Returns (xp, out_dtype, w, n, nan_count, mu, m2, m3, m4).
    m3 is None if up_to_order < 3; m4 is None if up_to_order < 4.
    Fills a NaN result vector and returns it as the final element too.
    """
    xp, arr, out_dtype = _prepare_input(x)
    n = arr.size
    w = int(window)

    require_positive(w, "window", kernel="rollmoments", kind="int")
    if w > n:
        return (
            xp,
            out_dtype,
            w,
            n,
            None,
            None,
            None,
            None,
            None,
            xp.full(n, xp.nan, dtype=out_dtype),
        )

    is_nan = xp.isnan(arr)
    zero_scalar = xp.asarray(0, dtype=out_dtype)
    x_safe = xp.where(is_nan, zero_scalar, arr)

    # Shift by first (non-NaN) value for stability.
    shift_val = float(x_safe[0]) if n > 0 else 0.0
    if not np.isfinite(shift_val):
        shift_val = 0.0
    shift = xp.asarray(shift_val, dtype=out_dtype)
    y = x_safe - shift

    zpad = xp.zeros(1, dtype=out_dtype)
    csy = xp.concatenate([zpad, xp.cumsum(y)])
    csy2 = xp.concatenate([zpad, xp.cumsum(y * y)])
    csy3 = xp.concatenate([zpad, xp.cumsum(y * y * y)]) if up_to_order >= 3 else None
    csy4 = xp.concatenate([zpad, xp.cumsum(y * y * y * y)]) if up_to_order >= 4 else None

    nan_int = is_nan.astype(np.int64)
    nan_zpad = xp.zeros(1, dtype=np.int64)
    csnan = xp.concatenate([nan_zpad, xp.cumsum(nan_int)])

    S1 = csy[w:] - csy[:-w]
    S2 = csy2[w:] - csy2[:-w]
    S3 = csy3[w:] - csy3[:-w] if csy3 is not None else None
    S4 = csy4[w:] - csy4[:-w] if csy4 is not None else None
    nnan = csnan[w:] - csnan[:-w]

    mu = S1 / w
    # 2nd central moment: sum((y - μ)²) / w = (S2 - w·μ²) / w = S2/w - μ²
    m2 = S2 / w - mu * mu
    m2 = xp.maximum(m2, zero_scalar)  # guard tiny negatives from FP rounding

    m3 = None
    m4 = None
    if S3 is not None:
        # 3rd central moment: (S3 - 3μ S2 + 2w μ³) / w
        m3 = (S3 - 3 * mu * S2 + 2 * w * mu * mu * mu) / w
    if S4 is not None:
        # Order >= 4 implies order >= 3, so S3 is guaranteed non-None here.
        # Explicit assertion narrows the type for Pylance/mypy.
        assert S3 is not None
        # 4th central moment: (S4 - 4μ S3 + 6μ² S2 - 3w μ⁴) / w
        m4 = (S4 - 4 * mu * S3 + 6 * mu * mu * S2 - 3 * w * mu * mu * mu * mu) / w
        m4 = xp.maximum(m4, zero_scalar)

    result = xp.full(n, xp.nan, dtype=out_dtype)
    return xp, out_dtype, w, n, nnan, mu, m2, m3, m4, result


def rollskew(x, window):
    """Rolling sample skewness (pandas convention: adjusted Fisher-Pearson).

    Requires `window >= 3`.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
    window : int
        Must be >= 3 for a defined skew; otherwise all NaN.

    Returns
    -------
    1D array, same length/backend/dtype.

    Examples
    --------
    >>> import numpy as np
    >>> rollskew(np.array([1.0, 2, 4, 8, 16]), 5)  # right-skewed → positive
    array([nan, nan, nan, nan, 1.375...])
    """
    xp, out_dtype, w, n, nnan, mu, m2, m3, _m4, result = _rolling_moments_setup(x, window, 3)
    if nnan is None:
        return result  # window > n or w <= 0 handled by setup
    if w < 3:
        return result  # all NaN
    assert m2 is not None and m3 is not None  # up_to_order=3 → guaranteed non-None

    valid_var = m2 > 0
    denom = xp.where(valid_var, m2**1.5, xp.asarray(1.0, dtype=out_dtype))
    g1 = m3 / denom
    # Bias correction: sqrt(w(w-1)) / (w-2)
    bias = np.sqrt(w * (w - 1)) / (w - 2)
    skew = bias * g1
    skew = xp.where(valid_var, skew, xp.asarray(xp.nan, dtype=out_dtype))

    valid = nnan == 0
    result[w - 1 :] = xp.where(valid, skew, xp.asarray(xp.nan, dtype=out_dtype))
    return result


def rollkurt(x, window):
    """Rolling excess kurtosis (pandas convention).

    Requires `window >= 4`. Returns Fisher/excess kurtosis (subtract 3 from
    the raw ratio), with bias correction.

    Parameters
    ----------
    x : 1D array (numpy or cupy)
    window : int
        Must be >= 4 for a defined kurtosis; otherwise all NaN.

    Examples
    --------
    >>> import numpy as np
    >>> rollkurt(np.random.default_rng(0).normal(size=100), 50)[-1]  # ~ 0 for N(0,1)
    """
    xp, out_dtype, w, n, nnan, mu, m2, m3, m4, result = _rolling_moments_setup(x, window, 4)
    if nnan is None:
        return result
    if w < 4:
        return result
    assert m2 is not None and m4 is not None  # up_to_order=4 → guaranteed non-None

    valid_var = m2 > 0
    m2_sq = m2 * m2
    denom = xp.where(valid_var, m2_sq, xp.asarray(1.0, dtype=out_dtype))
    ratio = m4 / denom  # non-excess kurtosis on m2 basis

    # Bias-corrected excess kurtosis (pandas formula):
    #   G2 = (w-1)/((w-2)(w-3)) * ((w+1)·ratio - 3(w-1))
    factor = (w - 1) / ((w - 2) * (w - 3))
    kurt = factor * ((w + 1) * ratio - 3 * (w - 1))
    kurt = xp.where(valid_var, kurt, xp.asarray(xp.nan, dtype=out_dtype))

    valid = nnan == 0
    result[w - 1 :] = xp.where(valid, kurt, xp.asarray(xp.nan, dtype=out_dtype))
    return result

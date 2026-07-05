"""Winsorize — cap values at chosen quantiles.

The one primitive every signals-desk re-implements. Given an array and
lower/upper quantiles, replace values below the `lo`-th quantile with
that quantile's value, and values above the `hi`-th quantile with the
upper quantile's value.

Two modes for 2D input:

- **`per_row=True`** (default) — CROSS-SECTIONAL winsorization: each
  row (each time slice) gets its own quantile boundaries. This is what
  you want for factor scores that span names at a given date.
- **`per_row=False`** — TIME-SERIES winsorization: each column (each
  name) gets its own quantile boundaries from its full history. Use
  for one-name-per-column noise clipping.

NaN handling: NaN is EXCLUDED from the quantile computation and
PRESERVED in the output (never clipped, never treated as a value).

Design: docs/kernels/signals/winsorize.md.
"""

from __future__ import annotations

import warnings

import numpy as np

from kuant._validation import require_probability, warn_kuant
from kuant.errors import KuantNumericWarning, KuantShapeError, KuantValueError


def winsorize(x, lo: float = 0.01, hi: float = 0.99, per_row: bool = True) -> np.ndarray:
    """Cap values at given quantiles. NaN preserved.

    Parameters
    ----------
    x : 1D or 2D array
        Input values. Floats; integers are promoted to float64 so NaN
        can be represented.
    lo : float in [0, 1], default 0.01
        Lower quantile. Values strictly below are clipped up to it.
    hi : float in [0, 1], default 0.99
        Upper quantile. Values strictly above are clipped down to it.
        Must be > `lo`.
    per_row : bool, default True
        Only applies to 2D input.
        - `True` — quantiles computed per row (cross-sectional).
        - `False` — quantiles computed per column (time-series).

    Returns
    -------
    np.ndarray of same shape and dtype (float) as `x`.

    Notes
    -----
    - NaN cells stay NaN in the output.
    - If a row/column is all-NaN, the corresponding output row/column
      is unchanged (all-NaN).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    >>> # Default (1%, 99%) → 100 is clipped down to the 99th pct.
    >>> winsorize(x, lo=0.0, hi=0.9).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.1]
    """
    require_probability(lo, "lo", kernel="winsorize")
    require_probability(hi, "hi", kernel="winsorize")
    if lo >= hi:
        raise KuantValueError(
            f"kuant.winsorize: 'lo' ({lo}) must be strictly less than 'hi' "
            f"({hi}).  [KE-VAL-RANGE]\n"
            f"  → Fix: pass lo < hi in [0, 1] — e.g. (0.01, 0.99) for the "
            f"standard 1st/99th-percentile cap"
        )
    if float(lo) > 0.25 or float(hi) < 0.75:
        warn_kuant(
            kernel="winsorize",
            code="KW-WINSORIZE-AGGRESSIVE-LIMITS",
            what=(
                f"limits (lo={lo}, hi={hi}) clip more than 50% of the "
                f"distribution to the boundary values"
            ),
            fix=(
                "typical values are (0.01, 0.99) or (0.05, 0.95); "
                "narrower limits closer to 0 and 1 keep the interior mass"
            ),
            category=KuantNumericWarning,
        )

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return _winsorize_1d(arr, lo, hi)
    if arr.ndim == 2:
        if per_row:
            out = arr.copy()
            for t in range(arr.shape[0]):
                out[t, :] = _winsorize_1d(arr[t, :], lo, hi)
            return out
        else:
            out = arr.copy()
            for c in range(arr.shape[1]):
                out[:, c] = _winsorize_1d(arr[:, c], lo, hi)
            return out
    raise KuantShapeError(
        f"kuant.winsorize: input must be 1D or 2D, got shape {arr.shape}.  "
        f"[KE-SHAPE-EXPECTED]\n"
        f"  → Fix: pass a 1D series or a 2D panel"
    )


def _winsorize_1d(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Winsorize a 1D array. NaN excluded from quantile computation."""
    finite_mask = np.isfinite(arr)
    if not bool(finite_mask.any()):
        return arr.copy()
    # nanquantile emits a warning on an all-NaN row; we've guarded above
    # but ignore anyway to be defensive.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo_val = float(np.nanquantile(arr, lo))
        hi_val = float(np.nanquantile(arr, hi))
    out = arr.copy()
    # Clip only finite entries; NaN stays NaN.
    clipped = np.clip(arr[finite_mask], lo_val, hi_val)
    out[finite_mask] = clipped
    return out


__all__ = ["winsorize"]

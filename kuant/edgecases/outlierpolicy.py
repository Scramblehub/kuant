"""Universal outlier detection with pluggable method.

Three standard outlier detectors surface here under one API:

- **`'mad'`** — Median Absolute Deviation. Robust to extreme values
  (doesn't get pulled by the outliers themselves). Threshold interpreted
  as a MAD multiple; default 3.0.
- **`'iqr'`** — Interquartile Range. Rejects entries outside
  `[Q1 - k*IQR, Q3 + k*IQR]`. Threshold `k` default 1.5 (Tukey fences).
- **`'zscore'`** — Standard z-score using mean/std. Fast but SENSITIVE
  to outliers (they inflate std, hiding themselves). Threshold default 3.

The output is a boolean mask (True = outlier), ready to compose with
`kuant.stats.winsorize` (planned for kuant.signals) or with any policy
from `kuant.edgecases.nanpolicies`.

Design: docs/kernels/edgecases/outlierpolicy.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import require_1d, require_positive, warn_kuant
from kuant.errors import KuantNumericWarning, KuantValueError

_ALLOWED_METHODS = ("mad", "iqr", "zscore")


def outlierpolicy(
    x,
    method: str = "mad",
    threshold: float | None = None,
) -> np.ndarray:
    """Detect outliers under a chosen method. Returns a boolean mask.

    Parameters
    ----------
    x : 1D array
        Input values. NaN is preserved as False in the output mask
        (NaN is neither an inlier nor an outlier — it's absent).
    method : {'mad', 'iqr', 'zscore'}, default 'mad'
        - `'mad'`: |x - median| > threshold * MAD (scale-consistent).
        - `'iqr'`: x < Q1 - threshold * IQR or x > Q3 + threshold * IQR.
        - `'zscore'`: |x - mean| > threshold * std.
    threshold : float, optional
        Method-specific threshold. Defaults:
          `'mad'` → 3.0 (approx 4.5 std for Gaussian)
          `'iqr'` → 1.5 (Tukey fences)
          `'zscore'` → 3.0

    Returns
    -------
    1D bool array of length `len(x)`
        True where the entry is an outlier.

    Warnings
    --------
    - `KuantNumericWarning` (`KW-OUTLIER-DEGENERATE`) if the method's
      scale statistic (MAD, IQR, std) is zero — every value is
      identical or the input is empty. Mask is returned as all-False.

    Notes
    -----
    - `'zscore'` is deliberately included but is the WORST choice
      when the outliers themselves are large: they pull mean and inflate
      std, hiding themselves inside a stretched interval. Prefer
      `'mad'` for financial return series.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2, 3, 4, 5, 100])   # 100 is the outlier
    >>> outlierpolicy(x, method='mad').tolist()
    [False, False, False, False, False, True]
    """
    if method not in _ALLOWED_METHODS:
        raise KuantValueError(
            f"kuant.outlierpolicy: 'method' must be one of "
            f"{_ALLOWED_METHODS}, got {method!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_METHODS}"
        )
    if threshold is None:
        threshold = {"mad": 3.0, "iqr": 1.5, "zscore": 3.0}[method]
    require_positive(threshold, "threshold", kernel="outlierpolicy")

    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="outlierpolicy")
    n = arr.size
    mask = np.zeros(n, dtype=bool)

    finite_mask = np.isfinite(arr)
    if not bool(finite_mask.any()):
        return mask
    values = arr[finite_mask]

    if method == "mad":
        med = float(np.median(values))
        deviations = np.abs(values - med)
        mad = float(np.median(deviations))
        if mad == 0.0:
            warn_kuant(
                kernel="outlierpolicy",
                code="KW-OUTLIER-DEGENERATE",
                what="MAD is zero — 50%+ of values equal the median",
                fix=(
                    "either the input is (near-)constant or the sample "
                    "is too small; consider 'iqr' or add more data"
                ),
                category=KuantNumericWarning,
            )
            return mask
        # 1.4826 scale factor makes MAD comparable to std under Gaussian.
        scaled = deviations / (mad * 1.4826)
        out_finite = scaled > threshold
    elif method == "iqr":
        q1, q3 = np.percentile(values, [25.0, 75.0])
        iqr = float(q3 - q1)
        if iqr == 0.0:
            warn_kuant(
                kernel="outlierpolicy",
                code="KW-OUTLIER-DEGENERATE",
                what="IQR is zero — 50% of values are identical",
                fix=(
                    "input is (near-)constant across the middle 50%; "
                    "consider 'mad' or add more diverse data"
                ),
                category=KuantNumericWarning,
            )
            return mask
        lo = q1 - threshold * iqr
        hi = q3 + threshold * iqr
        out_finite = (values < lo) | (values > hi)
    else:  # zscore
        mu = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if sd == 0.0:
            warn_kuant(
                kernel="outlierpolicy",
                code="KW-OUTLIER-DEGENERATE",
                what="std is zero — every finite value is identical",
                fix="input is constant; nothing can be an outlier",
                category=KuantNumericWarning,
            )
            return mask
        out_finite = (np.abs(values - mu) / sd) > threshold

    mask[finite_mask] = out_finite
    return mask


__all__ = ["outlierpolicy"]

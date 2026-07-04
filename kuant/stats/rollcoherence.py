"""Rolling magnitude-squared coherence in a target frequency band.

For each trailing window of length `window`, compute the Welch magnitude-
squared coherence between `x` and `y`, then return the mean value inside
the frequency band `[band_lo, band_hi]` (cycles per sample).

Uses scipy.signal.coherence for the spectral estimate.

Design: docs/kernels/stats/rollcoherence.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import (
    require_dep,
    require_equal_length,
    require_positive,
    require_range,
)
from kuant.errors import KuantValueError


def rollcoherence(
    x,
    y,
    window: int,
    nperseg: int | None = None,
    band: tuple[float, float] = (0.0, 0.5),
    fs: float = 1.0,
):
    """Rolling coherence between two 1D series, integrated over a band.

    Parameters
    ----------
    x, y : 1D arrays
        Input series (must have equal length).
    window : int
        Trailing window length.
    nperseg : int, optional
        Welch segment length. Defaults to `window // 2`.
    band : tuple(lo, hi), default (0.0, 0.5)
        Frequency band (cycles / sample) over which to average |C_xy|²
    fs : float, default 1.0
        Sampling frequency. If your data is daily, keep 1.0 and interpret
        band in cycles per day.

    Returns
    -------
    1D np.ndarray of length == len(x)
        Mean |C_xy|² in the band at each anchor. NaN in warm-up region.

    Raises
    ------
    ImportError
        If scipy isn't installed.
    ValueError
        On mismatched shapes or invalid window.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = 0.5 * x + rng.standard_normal(500)
    >>> c = rollcoherence(x, y, window=200, band=(0.0, 0.3))
    >>> np.all(np.isnan(c[:199]))
    True
    """
    try:
        from scipy.signal import coherence
    except ImportError as e:
        require_dep(
            "scipy",
            kernel="rollcoherence",
            install="pip install scipy",
            cause=e,
        )

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    require_equal_length(x_arr, "x", y_arr, "y", kernel="rollcoherence")

    n = x_arr.size
    w = int(window)
    require_positive(w, "window", kernel="rollcoherence", kind="int")
    require_positive(fs, "fs", kernel="rollcoherence")
    lo_band, hi_band = band
    require_range(lo_band, "band[0]", kernel="rollcoherence", lo=0.0, hi=fs / 2)
    require_range(hi_band, "band[1]", kernel="rollcoherence", lo=0.0, hi=fs / 2)
    if lo_band >= hi_band:
        raise KuantValueError(
            f"kuant.rollcoherence: 'band' lo={lo_band} must be strictly less "
            f"than hi={hi_band}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pass (lo, hi) with lo < hi in cycles/sample "
            f"(Nyquist = fs/2 = {fs / 2})"
        )
    if nperseg is not None:
        require_positive(nperseg, "nperseg", kernel="rollcoherence", kind="int")
    if nperseg is None:
        nperseg = max(8, w // 2)

    lo, hi = band
    result = np.full(n, np.nan)

    for t in range(w - 1, n):
        xw = x_arr[t - w + 1 : t + 1]
        yw = y_arr[t - w + 1 : t + 1]
        if not (np.all(np.isfinite(xw)) and np.all(np.isfinite(yw))):
            continue
        try:
            f, Cxy = coherence(xw, yw, fs=fs, nperseg=min(nperseg, w))
        except ValueError:
            continue
        mask = (f >= lo) & (f < hi)
        if mask.any():
            result[t] = float(np.mean(Cxy[mask]))

    return result

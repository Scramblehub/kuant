"""Rolling magnitude-squared coherence in a target frequency band.

For each trailing window of length `window`, compute the Welch magnitude-
squared coherence between `x` and `y`, then return the mean value inside
the frequency band `[band_lo, band_hi]` (cycles per sample).

Uses scipy.signal.coherence for the spectral estimate.

Design: docs/kernels/stats/rollcoherence.md.
"""

from __future__ import annotations

import numpy as np


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
        raise ImportError(
            "kuant.stats.rollcoherence requires scipy.signal.coherence. "
            "scipy is a hard dep of kuant, so this should always be available; "
            "install with: pip install scipy"
        ) from e

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.size != y_arr.size:
        raise ValueError(f"x and y must have equal length; got {x_arr.size} vs {y_arr.size}")

    n = x_arr.size
    w = int(window)
    if w <= 0:
        raise ValueError(f"window must be positive, got {w}")
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

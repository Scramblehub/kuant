"""Rolling Hurst exponent.

At each anchor `t >= window - 1`, fit an R/S Hurst exponent on the
trailing `window` samples. Useful when the underlying series may
exhibit regime-varying self-similarity — e.g. periods of persistence
interleaved with periods of mean-reversion.

Composes `hurstrs` in a trailing-window loop.

Design: docs/kernels/stats/rollhurst.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError

from .hurstrs import hurstrs

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _to_numpy(r):
    if isinstance(r, _CUPY_NDARRAY):
        return r.get()
    return np.asarray(r, dtype=np.float64)


def rollhurst(r, window: int = 252, min_w: int = 8, n_windows: int = 8) -> np.ndarray:
    """Rolling Hurst exponent on trailing windows.

    Parameters
    ----------
    r : 1D array
        Input series.
    window : int, default 252
        Trailing-window length (bars). At each `t`, the R/S fit uses
        `r[t - window + 1 : t + 1]`.
    min_w : int, default 8
        R/S inner minimum sub-window. Kept small so short outer
        windows still have enough log-log points to fit.
    n_windows : int, default 8
        R/S inner window count. Small default balances speed vs
        regression stability inside each trailing window.

    Returns
    -------
    1D numpy.ndarray, length == len(r)
        `H_t`; NaN for `t < window - 1` or when the inner fit fails.

    Notes
    -----
    This is an O(n · window) loop and is not the fastest way to
    compute a rolling exponent. For very long series or heavy-use
    scans, cache the result and reuse.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(1000)
    >>> H_t = rollhurst(r, window=250)
    >>> np.isnan(H_t[:249]).all()
    True
    """
    r_np = _to_numpy(r)
    require_1d(r_np, "r", kernel="rollhurst")
    if window < 4 * min_w:
        raise KuantValueError(
            f"kuant.rollhurst: 'window' ({window}) must be at least 4 * "
            f"min_w ({4 * min_w}) for R/S regression to have enough scale "
            f"points.  [KE-VAL-RANGE]\n"
            f"  → Fix: raise `window` to at least {4 * min_w}, or lower "
            f"`min_w`"
        )

    n = r_np.size
    H = np.full(n, np.nan)
    for t in range(window - 1, n):
        segment = r_np[t - window + 1 : t + 1]
        try:
            result = hurstrs(segment, min_w=min_w, n_windows=n_windows)
            H[t] = result.H
        except ValueError:
            continue
    return H

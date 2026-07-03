"""Rolling Hill tail index over trailing windows.

At each anchor `t >= window - 1`, apply `tailindex` to the trailing
`window` samples. Useful as a regime signal: rising ξ_t indicates
fattening left tail.

Design: docs/kernels/stats/rolltailindex.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import require_positive

from .tailindex import tailindex


def rolltailindex(x, window: int, k_frac: float = 0.10, min_k: int = 10):
    """Rolling Hill tail index.

    Parameters
    ----------
    x : 1D array
        Positive values (typically loss magnitudes) OR raw returns
        (in which case pass `-returns.clip(max=0)` for the left-tail).
    window : int
        Trailing window size.
    k_frac : float, default 0.10
        Fraction of samples in each window used for the tail.
    min_k : int, default 10
        Minimum tail size; windows with fewer valid samples → NaN.

    Returns
    -------
    1D np.ndarray, length == len(x)
        ξ_t; NaN for `t < window - 1`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> losses = rng.pareto(2.0, size=1000) + 1
    >>> xi_t = rolltailindex(losses, window=250)
    >>> np.isfinite(xi_t[-1])
    True
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = arr.size
    w = int(window)
    require_positive(w, "window", kernel="rolltailindex", kind="int")

    result = np.full(n, np.nan)
    for t in range(w - 1, n):
        result[t] = tailindex(arr[t - w + 1 : t + 1], k_frac=k_frac, min_k=min_k)
    return result

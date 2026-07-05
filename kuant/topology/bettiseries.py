"""Rolling Betti-number time series from a scalar input.

The n-th Betti number `b_n` counts the number of n-dimensional
topological features (`b_0` = connected components, `b_1` = loops,
`b_2` = voids, ...) in a filtered simplicial complex. Persistence
returns birth/death PAIRS — Betti is a COUNT.

`bettiseries` slides a window across a 1D input, builds the Takens
time-delay embedding inside each window, runs persistence, and counts
features whose persistence exceeds a threshold. The result is a 1D
signal whose spikes localize regime changes (topology gain) or
consolidations (topology loss) in the underlying dynamics.

Reference application: b_1 spikes correlate with regime transitions
in some quant literature — the state space's attractor sprouts loops
as the dynamics move between regions.

Design: docs/kernels/topology/bettiseries.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import require_1d, require_nonnegative, require_positive
from kuant.errors import KuantValueError

from .persistenthomology import persistenthomology


def bettiseries(
    x,
    window: int,
    dim: int = 1,
    embedding_dim: int = 3,
    delay: int = 1,
    min_persistence: float = 0.0,
    stride: int = 1,
    max_edge_length: float | None = None,
) -> np.ndarray:
    """Rolling Betti-number count over a trailing window.

    At each anchor `t >= window - 1`, embed `x[t-window+1 : t+1]` via
    Takens, run persistent homology, and count features at `dim` with
    persistence >= `min_persistence`. Warm-up entries are NaN.

    Parameters
    ----------
    x : 1D array
        Input series.
    window : int
        Trailing window size (number of samples fed to the embedding).
    dim : int, default 1
        Homology dimension to count (0 = components, 1 = loops, ...).
    embedding_dim : int, default 3
        Takens embedding dimension.
    delay : int, default 1
        Takens delay τ.
    min_persistence : float, default 0.0
        Only count features with `death - birth >= min_persistence`.
        Set > 0 to filter noise; 0 counts everything ripser returns.
    stride : int, default 1
        Compute Betti at every `stride`-th anchor; interpolate NaN
        between. Set > 1 when the persistence step is the bottleneck.
    max_edge_length : float, optional
        Truncation ε for the Rips filtration inside each window.

    Returns
    -------
    1D np.ndarray, length == len(x)
        Betti-`dim` count per anchor. NaN in warm-up
        (indices 0..window-2) and at anchors skipped by stride.

    Notes
    -----
    Compute cost per anchor is dominated by ripser on a `~window`-point
    cloud. For window=100 the per-anchor cost is ~1-10 ms on CPU, so a
    dense series of length 10_000 with stride=1 takes minutes. Use
    stride > 1 for long series where you only want low-frequency
    changes.

    Infinite-death features (persistence = ∞) always count when
    `min_persistence` is finite. To exclude them, set
    `min_persistence = np.inf` (which then also excludes everything
    else — so use a helper post-filter instead).

    Examples
    --------
    >>> import numpy as np                                                     # doctest: +SKIP
    >>> t = np.linspace(0, 10 * np.pi, 500)                                    # doctest: +SKIP
    >>> x = np.sin(t)                                                          # doctest: +SKIP
    >>> b1 = bettiseries(x, window=100, dim=1, min_persistence=0.2)            # doctest: +SKIP
    >>> np.isnan(b1[:99]).all()                                                # doctest: +SKIP
    True
    >>> (b1[99:] >= 1).mean() > 0.5                                            # sine attractor is a loop
    True
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="bettiseries")
    require_positive(window, "window", kernel="bettiseries", kind="int")
    require_positive(stride, "stride", kernel="bettiseries", kind="int")
    require_nonnegative(min_persistence, "min_persistence", kernel="bettiseries")

    n = arr.size
    w = int(window)
    s = int(stride)
    if w > n:
        raise KuantValueError(
            f"kuant.bettiseries: 'window' ({w}) is larger than len(x) "
            f"({n}); no anchor can be evaluated.  [KE-VAL-RANGE]\n"
            f"  → Fix: lower window or provide a longer series"
        )
    out = np.full(n, np.nan)

    for t in range(w - 1, n, s):
        segment = arr[t - w + 1 : t + 1]
        # NaN in the segment breaks the embedding; return NaN for that anchor.
        if not np.all(np.isfinite(segment)):
            continue
        d = persistenthomology(
            segment,
            dim=dim,
            embedding_dim=embedding_dim,
            delay=delay,
            max_edge_length=max_edge_length,
        )
        out[t] = float(d.n_features(dim, min_persistence=min_persistence))

    return out


__all__ = ["bettiseries"]

"""Cross-sectional dispersion-collapse signal.

Given a matrix of returns `R[t, i]` (rows = bars, cols = names), compute
the cross-sectional dispersion at each bar and flag bars where
dispersion has been persistently low.

    dispersion_t = std_over_names(R[t, :])
    collapsed_t  = 1 if dispersion_t < quantile(dispersion history, q)
                       for `n_consecutive` consecutive bars
                   0 otherwise

Motivation. When cross-sectional dispersion collapses — all names
moving in lock-step — the market's factor structure is dominated by
a single macro driver and idiosyncratic alpha capacity is small.
Historically this pattern precedes some regime shifts and reduces
short-horizon carry on relative-value strategies.

Not persistent-homology in the H0/H1 sense; the "topology" bucket
label reflects that this is a shape metric on the returns
distribution rather than a level or a moment. Preserved as reference
because it fell out of prior research and is trivially cheap.

Design: docs/kernels/topology/dispersioncollapse.md.
"""

from __future__ import annotations

import warnings

import numpy as np

from kuant._validation import (
    require_2d,
    require_positive,
    require_probability,
)


def dispersioncollapse(
    returns_matrix,
    window: int = 63,
    quantile: float = 0.20,
    n_consecutive: int = 5,
    return_dispersion: bool = False,
):
    """Boolean signal that fires on persistent low cross-sectional dispersion.

    Parameters
    ----------
    returns_matrix : 2D array
        Returns with shape `(n_bars, n_names)`. Each row is one bar.
        NaN in a cell is treated as "no observation for that name at
        that bar" — the dispersion at bar `t` uses only names with a
        finite return there.
    window : int, default 63
        Trailing window over which the dispersion QUANTILE is computed.
        The comparison is per-bar: bar `t`'s dispersion is compared to
        the `quantile`-th percentile of the previous `window` bars'
        dispersions.
    quantile : float in [0, 1], default 0.20
        Percentile of the dispersion history that marks "low".
    n_consecutive : int, default 5
        Minimum number of consecutive low-dispersion bars required to
        fire the signal.
    return_dispersion : bool, default False
        If True, also return the raw dispersion series.

    Returns
    -------
    collapsed : 1D bool array of length `n_bars`
        True at bars where the last `n_consecutive` bars (inclusive)
        were all below the trailing-window `quantile`. False in
        warm-up (`t < window + n_consecutive - 2`) and at bars that
        don't satisfy the persistence rule.
    dispersion : 1D float array, optional
        The per-bar cross-sectional standard deviation. Returned only
        if `return_dispersion` is True. NaN where fewer than 2 names
        had a finite return at that bar.

    Notes
    -----
    Uses `np.nanstd(ddof=1)` for the cross-sectional dispersion so
    NaN cells are ignored per-bar. Requires at least 2 non-NaN
    observations per bar to define dispersion; fewer → NaN, which
    never triggers the collapse condition.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> normal = rng.normal(0, 0.02, size=(500, 20))                              # noisy
    >>> lockstep = np.tile(rng.normal(0, 0.02, size=(50, 1)), (1, 20))            # all names move together
    >>> R = np.vstack([normal[:200], lockstep, normal[250:]])
    >>> c = dispersioncollapse(R, window=63, quantile=0.20, n_consecutive=5)
    >>> c[210:250].mean() > c[:200].mean()                                        # signal fires in lockstep block
    True
    """
    arr = np.asarray(returns_matrix, dtype=np.float64)
    require_2d(arr, "returns_matrix", kernel="dispersioncollapse")
    require_positive(window, "window", kernel="dispersioncollapse", kind="int")
    require_probability(quantile, "quantile", kernel="dispersioncollapse")
    require_positive(n_consecutive, "n_consecutive", kernel="dispersioncollapse", kind="int")

    n_bars, n_names = arr.shape

    # Per-bar cross-sectional dispersion. nanstd emits a "Degrees of
    # freedom <= 0" RuntimeWarning when a row has fewer than 2 finite
    # observations and returns NaN — the semantics we want. Silence the
    # noise so users don't see it.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        dispersion = np.nanstd(arr, axis=1, ddof=1)
    # nanstd returns 0 (not NaN) for all-NaN rows on some numpy versions.
    # Force NaN when fewer than 2 finite observations were available.
    n_finite = np.isfinite(arr).sum(axis=1)
    dispersion[n_finite < 2] = np.nan

    # Trailing-window quantile — one anchor per bar. O(n_bars · window log window)
    # via sliding view + np.nanquantile.
    is_low = np.zeros(n_bars, dtype=bool)
    w = int(window)
    for t in range(w - 1, n_bars):
        past = dispersion[t - w + 1 : t + 1]
        past_clean = past[np.isfinite(past)]
        if past_clean.size < 2:
            continue
        threshold = float(np.nanquantile(past_clean, quantile))
        cur = dispersion[t]
        if np.isfinite(cur) and cur < threshold:
            is_low[t] = True

    # Persistence: True only if the trailing `n_consecutive` bars are all low.
    k = int(n_consecutive)
    collapsed = np.zeros(n_bars, dtype=bool)
    if k <= 1:
        collapsed[:] = is_low
    else:
        # Rolling AND over a length-k window via convolution.
        counts = np.convolve(is_low.astype(np.int64), np.ones(k, dtype=np.int64), mode="valid")
        collapsed[k - 1 :] = counts == k

    if return_dispersion:
        return collapsed, dispersion
    return collapsed


__all__ = ["dispersioncollapse"]

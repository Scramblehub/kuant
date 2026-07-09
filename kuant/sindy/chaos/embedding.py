"""Time-delay embedding helpers: mutual information + false nearest neighbors.

Both are the standard tools for picking the embedding parameters
`(tau, m)` that CCM, Lyapunov, and correlation dimension all depend on:

- **mutualinfo(x, y=None, lag, ...)** picks the delay `tau`. The
  convention is "first minimum of the auto-mutual-information curve
  vs lag." That minimum is where successive time-delay coordinates are
  as informationally-independent as possible.
- **falsenearest(x, tau, max_dim, ...)** picks the embedding dimension
  `m`. The false-nearest-neighbors fraction drops sharply once `m` is
  large enough to fully unfold the attractor; picking the first `m`
  where the fraction is below a small threshold (default 5%) is the
  Kennel-Brown-Abarbanel heuristic.

Both estimators use histogram / k-NN based methods that are cheap
enough for the small samples typical in financial time series
(hundreds to low tens of thousands of observations).

Design: docs/kernels/sindy/chaos/embedding.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
from kuant.errors import KuantValueError


# ---------- mutual information ----------------------------------------


@dataclass
class MutualInfoResult:
    """Auto-mutual-information curve vs lag.

    Attributes
    ----------
    lags : 1D np.ndarray
    mi : 1D np.ndarray
        Mutual information (nats) at each lag.
    suggested_tau : int
        Lag at the first local minimum, or `1` if no minimum found in
        the range.
    """

    lags: np.ndarray
    mi: np.ndarray
    suggested_tau: int

    def summary(self) -> str:
        return (
            "=== MutualInfoResult ===\n"
            f"lags tested:     {self.lags[0]} .. {self.lags[-1]}\n"
            f"MI at lag 1:     {self.mi[0]:.4f} nats\n"
            f"MI at lag {self.lags[-1]}:  {self.mi[-1]:.4f} nats\n"
            f"suggested tau:   {self.suggested_tau}"
        )


def _histogram_mi(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Shannon MI in nats from a joint histogram."""
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    total = joint.sum()
    if total <= 0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    # Compute ratio only on cells with strictly positive joint AND
    # marginals, avoiding division-by-zero warnings.
    marginal_ok = (px @ py) > 0
    nonzero = (pxy > 0) & marginal_ok
    if not np.any(nonzero):
        return 0.0
    ratio = pxy[nonzero] / (px * py)[nonzero]
    return float(np.sum(pxy[nonzero] * np.log(ratio)))


def mutualinfo(
    x,
    y=None,
    *,
    lag=1,
    bins: int = 32,
    max_lag: int = 32,
) -> MutualInfoResult | float:
    """Mutual information via histogram binning.

    Two calling modes:

    1. **Auto-MI curve**: pass `x` only. Returns a `MutualInfoResult`
       with MI vs lag from 1 to `max_lag`. `suggested_tau` is the first
       local minimum.
    2. **Cross-MI scalar**: pass `x` and `y`. Returns MI between `x` and
       `y[lag:]` shifted-appropriately, as a `float`.

    Parameters
    ----------
    x : 1D array
    y : 1D array, optional
        If None, computes auto-MI curve. Same length as x.
    lag : int, default 1
        The lag at which to compute cross-MI (mode 2 only).
    bins : int, default 32
        Histogram bins per marginal.
    max_lag : int, default 32
        Max lag to include in the auto-MI curve (mode 1 only).

    Returns
    -------
    MutualInfoResult (mode 1) or float (mode 2)

    References
    ----------
    Fraser & Swinney 1986, "Independent coordinates for strange
    attractors from mutual information."
    """
    arr_x = np.asarray(x, dtype=np.float64)
    require_1d(arr_x, "x", kernel="mutualinfo")
    require_positive(bins, "bins", kernel="mutualinfo", kind="int")
    finite_x = arr_x[np.isfinite(arr_x)]
    if finite_x.size < 32:
        raise KuantValueError(
            f"kuant.mutualinfo: only {finite_x.size} finite values; "
            f"need at least 32 for meaningful MI.  [KE-VAL-MIN-CLEAN]\n"
            f"  → Fix: provide more data"
        )
    if y is not None:
        arr_y = np.asarray(y, dtype=np.float64)
        require_1d(arr_y, "y", kernel="mutualinfo")
        if arr_x.size != arr_y.size:
            raise KuantValueError(
                f"kuant.mutualinfo: 'x' and 'y' must be the same length; "
                f"got {arr_x.size} and {arr_y.size}.  "
                f"[KE-SHAPE-EQUAL-LEN]"
            )
        require_positive(lag, "lag", kernel="mutualinfo", kind="int")
        # Cross-MI at the given lag.
        if lag >= arr_x.size:
            raise KuantValueError(
                f"kuant.mutualinfo: 'lag' ({lag}) >= len(x) " f"({arr_x.size}).  [KE-VAL-RANGE]"
            )
        xL = arr_x[: arr_x.size - lag]
        yL = arr_y[lag:]
        mask = np.isfinite(xL) & np.isfinite(yL)
        return _histogram_mi(xL[mask], yL[mask], bins)
    # Auto-MI curve.
    require_range(max_lag, "max_lag", kernel="mutualinfo", lo=1, hi=1e9)
    if max_lag >= arr_x.size // 2:
        raise KuantValueError(
            f"kuant.mutualinfo: 'max_lag' ({max_lag}) must be less than "
            f"len(x)/2 ({arr_x.size // 2}).  [KE-VAL-RANGE]"
        )
    lags = np.arange(1, int(max_lag) + 1)
    mi = np.empty(lags.size, dtype=np.float64)
    for i, k in enumerate(lags):
        xL = arr_x[: arr_x.size - k]
        xR = arr_x[k:]
        mask = np.isfinite(xL) & np.isfinite(xR)
        mi[i] = _histogram_mi(xL[mask], xR[mask], bins)
    # First local minimum.
    suggested = 1
    for i in range(1, mi.size - 1):
        if mi[i] < mi[i - 1] and mi[i] < mi[i + 1]:
            suggested = int(lags[i])
            break
    return MutualInfoResult(lags=lags, mi=mi, suggested_tau=suggested)


# ---------- false nearest neighbors -----------------------------------


@dataclass
class FalseNearestResult:
    """FNN fraction vs embedding dimension.

    Attributes
    ----------
    dims : 1D np.ndarray
    fnn : 1D np.ndarray
        Fraction of false nearest neighbors at each embedding dim, in
        [0, 1].
    suggested_m : int
        First dim where `fnn <= threshold`, or `max_dim` if none.
    threshold : float
        The FNN threshold used to pick suggested_m.
    """

    dims: np.ndarray
    fnn: np.ndarray
    suggested_m: int
    threshold: float

    def summary(self) -> str:
        return (
            "=== FalseNearestResult ===\n"
            f"dims tested:     {self.dims[0]} .. {self.dims[-1]}\n"
            f"FNN at m=1:      {self.fnn[0]:.4f}\n"
            f"FNN at m={self.dims[-1]}:      {self.fnn[-1]:.4f}\n"
            f"threshold:       {self.threshold}\n"
            f"suggested m:     {self.suggested_m}"
        )


def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Time-delay embedding: shape (N - (m-1)*tau, m)."""
    N = x.size - (m - 1) * tau
    out = np.empty((N, m), dtype=np.float64)
    for j in range(m):
        out[:, j] = x[j * tau : j * tau + N]
    return out


def falsenearest(
    x,
    *,
    tau: int = 1,
    max_dim: int = 10,
    r_tol: float = 15.0,
    threshold: float = 0.05,
) -> FalseNearestResult:
    """False nearest neighbors fraction vs embedding dimension.

    Parameters
    ----------
    x : 1D array
    tau : int, default 1
        Embedding delay. Use `mutualinfo(x).suggested_tau` if unsure.
    max_dim : int, default 10
    r_tol : float, default 15.0
        Kennel-Brown-Abarbanel tolerance ratio. Two neighbors are
        declared "false" if their (m+1)-th coordinate ratio exceeds
        `r_tol`. Kennel et al. suggest 10-30.
    threshold : float, default 0.05
        FNN fraction below which we consider the embedding dimension
        sufficient.

    Returns
    -------
    FalseNearestResult

    References
    ----------
    Kennel, Brown & Abarbanel 1992, "Determining embedding dimension
    for phase-space reconstruction using a geometrical construction."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="falsenearest")
    require_positive(tau, "tau", kernel="falsenearest", kind="int")
    require_range(max_dim, "max_dim", kernel="falsenearest", lo=1, hi=50)
    require_range(r_tol, "r_tol", kernel="falsenearest", lo=1.0, hi=1e6)
    require_range(threshold, "threshold", kernel="falsenearest", lo=0.0, hi=1.0)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size < 100:
        raise KuantValueError(
            f"kuant.falsenearest: only {arr.size} finite values; need "
            f"at least 100 for a stable FNN estimate.  "
            f"[KE-VAL-MIN-CLEAN]"
        )
    dims = np.arange(1, int(max_dim) + 1)
    fnn = np.empty(dims.size, dtype=np.float64)
    for i, m in enumerate(dims):
        # Embed at m and at m+1; find each point's nearest neighbor in
        # the m-dim embedding and check whether it stays close at m+1.
        embed_m = _embed(arr, int(m), int(tau))
        embed_m1 = _embed(arr, int(m) + 1, int(tau))
        # Truncate the m-dim embedding to match the m+1-dim length.
        N = embed_m1.shape[0]
        embed_m = embed_m[:N]
        if N < 10:
            fnn[i] = 1.0
            continue
        # Naive brute-force nearest neighbor (O(N^2)). Fine for the
        # small N (~hundreds to thousands) typical here.
        d2 = np.sum((embed_m[:, None, :] - embed_m[None, :, :]) ** 2, axis=-1)
        # Exclude self.
        np.fill_diagonal(d2, np.inf)
        nn_idx = np.argmin(d2, axis=1)
        nn_d = np.sqrt(d2[np.arange(N), nn_idx])
        # Distance at m+1: the extra coordinate is arr shifted by m*tau.
        extra_i = arr[m * int(tau) : m * int(tau) + N]
        extra_nn = extra_i[nn_idx]
        d_extra = np.abs(extra_i - extra_nn)
        # Kennel criterion: false if d_extra / nn_d > r_tol.
        # Guard nn_d == 0 by treating those pairs as "true" (not false).
        safe = nn_d > 0
        ratios = np.zeros(N, dtype=np.float64)
        ratios[safe] = d_extra[safe] / nn_d[safe]
        n_false = int(np.sum(ratios > r_tol))
        fnn[i] = n_false / N
    # Pick suggested m.
    suggested_m = int(dims[-1])
    for i, f in enumerate(fnn):
        if f <= threshold:
            suggested_m = int(dims[i])
            break
    return FalseNearestResult(
        dims=dims,
        fnn=fnn,
        suggested_m=suggested_m,
        threshold=float(threshold),
    )


__all__ = [
    "MutualInfoResult",
    "FalseNearestResult",
    "mutualinfo",
    "falsenearest",
    "_embed",
]

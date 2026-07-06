"""Convergent cross-mapping for coupled deterministic systems.

CCM (Sugihara et al 2012) tests for causality in coupled deterministic
systems by asking: can `y_t` be predicted from `x`'s shadow manifold?
If so, `x` and `y` are dynamically coupled and `y`'s information is
present in `x`'s history. The "convergence" that names the method is
the observation that prediction skill `rho(L)` grows with library
size `L` and saturates when `x -> y` is a true causal link. If skill
does not grow with `L`, the apparent correlation is not a dynamical
coupling.

CCM sidesteps two problems that trip Granger causality:
- Nonlinear couplings.
- Common-driver confounds (both series share a hidden third driver).

It also has failure modes: CCM assumes deterministic dynamics on a
low-dim attractor. Noisy stochastic data can give false positives at
small L that resolve at large L, so the convergence check is
important. Do NOT skip it.

Design: docs/kernels/sindy/chaos/ccm.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
from kuant.errors import KuantValueError
from kuant.sindy.chaos.embedding import _embed


@dataclass
class CCMResult:
    """Convergent cross-mapping predictability curves.

    Attributes
    ----------
    lib_sizes : 1D np.ndarray
        Library sizes tested.
    rho_xy : 1D np.ndarray
        Prediction skill (Pearson correlation) of `y` from `x`'s shadow
        manifold. Rising with library size implies `y` -> `x` (y's
        information lives in x's history, so x can predict y).
    rho_yx : 1D np.ndarray
        Same in the reverse direction. Rising implies `x` -> `y`.
    convergence_xy : bool
        Whether rho_xy(largest L) exceeds rho_xy(smallest L) by more
        than `convergence_threshold`.
    convergence_yx : bool
    convergence_threshold : float
    embed_dim : int
    embed_tau : int
    """

    lib_sizes: np.ndarray
    rho_xy: np.ndarray
    rho_yx: np.ndarray
    convergence_xy: bool
    convergence_yx: bool
    convergence_threshold: float
    embed_dim: int
    embed_tau: int

    def summary(self) -> str:
        conv_xy = "yes" if self.convergence_xy else "no"
        conv_yx = "yes" if self.convergence_yx else "no"
        return (
            "=== CCMResult ===\n"
            f"embed dim / tau:            {self.embed_dim} / {self.embed_tau}\n"
            f"lib sizes:                  {self.lib_sizes[0]} .. "
            f"{self.lib_sizes[-1]} (n={self.lib_sizes.size})\n"
            f"rho_xy at max L:            {self.rho_xy[-1]:+.4f}  "
            f"(convergent: {conv_xy})\n"
            f"rho_yx at max L:            {self.rho_yx[-1]:+.4f}  "
            f"(convergent: {conv_yx})\n"
            f"convergence threshold:      {self.convergence_threshold:.3f}"
        )


def _cross_map_score(shadow: np.ndarray, target: np.ndarray, lib_size: int, seed: int) -> float:
    """Predict `target` from `shadow` via simplex projection.

    `shadow` shape (N, m), `target` shape (N,). Uses a random subset of
    `lib_size` points as the library; predicts every point (including
    non-library rows) via a k=(m+1)-nearest-neighbor weighted mean.
    Returns Pearson correlation between predictions and target.
    """
    rng = np.random.default_rng(seed)
    N, m = shadow.shape
    if lib_size > N:
        lib_size = N
    lib_idx = rng.choice(N, size=int(lib_size), replace=False)
    lib = shadow[lib_idx]
    lib_targets = target[lib_idx]
    k = m + 1

    # For every point, find its k-NN in lib (excluding itself if in lib).
    preds = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        # Distances to library points.
        d = np.linalg.norm(lib - shadow[i], axis=1)
        # Exclude exact self if `i` is in lib.
        if i in lib_idx:
            d[np.where(lib_idx == i)[0][0]] = np.inf
        order = np.argsort(d)
        nn = order[:k]
        d_nn = d[nn]
        if not np.isfinite(d_nn[0]):
            continue
        # Simplex weights: exp(-d/d_min).
        d_min = d_nn[0] if d_nn[0] > 1e-15 else 1e-15
        w = np.exp(-d_nn / d_min)
        w /= w.sum()
        preds[i] = float(np.sum(w * lib_targets[nn]))

    mask = np.isfinite(preds)
    if int(mask.sum()) < 4:
        return float("nan")
    a = preds[mask] - preds[mask].mean()
    b = target[mask] - target[mask].mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom < 1e-15:
        return float("nan")
    return float(np.sum(a * b) / denom)


def ccm(
    x,
    y,
    *,
    tau: int = 1,
    m: int = 5,
    lib_sizes: list | None = None,
    n_seeds: int = 5,
    convergence_threshold: float = 0.1,
    seed: int = 0,
) -> CCMResult:
    """Convergent cross-mapping between two series.

    Parameters
    ----------
    x, y : 1D arrays of equal length
    tau : int, default 1
        Embedding delay.
    m : int, default 5
        Embedding dimension.
    lib_sizes : list of int, optional
        Library sizes to test. Default: 5 log-spaced sizes from
        `(m+1) * 2` to `N`.
    n_seeds : int, default 5
        Number of random library draws per library size. Result is the
        mean rho across draws.
    convergence_threshold : float, default 0.10
        Minimum increase in rho from smallest to largest library size
        to declare "convergent" causality.
    seed : int, default 0

    Returns
    -------
    CCMResult

    References
    ----------
    Sugihara et al 2012, "Detecting causality in complex ecosystems."
    """
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    require_1d(arr_x, "x", kernel="ccm")
    require_1d(arr_y, "y", kernel="ccm")
    if arr_x.size != arr_y.size:
        raise KuantValueError(
            f"kuant.ccm: 'x' and 'y' must be the same length; got "
            f"{arr_x.size} and {arr_y.size}.  [KE-SHAPE-EQUAL-LEN]"
        )
    require_positive(tau, "tau", kernel="ccm", kind="int")
    require_range(m, "m", kernel="ccm", lo=2, hi=50)
    require_positive(n_seeds, "n_seeds", kernel="ccm", kind="int")
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x = arr_x[mask]
    arr_y = arr_y[mask]
    if arr_x.size < 200:
        raise KuantValueError(
            f"kuant.ccm: only {arr_x.size} clean rows; need at least "
            f"200 for stable cross-mapping.  [KE-VAL-MIN-CLEAN]"
        )
    Ex = _embed(arr_x, int(m), int(tau))
    Ey = _embed(arr_y, int(m), int(tau))
    # Both embeddings have the same number of rows.
    N = Ex.shape[0]
    ty = arr_y[(int(m) - 1) * int(tau) :][:N]
    tx = arr_x[(int(m) - 1) * int(tau) :][:N]

    if lib_sizes is None:
        min_lib = int(m) + 1
        lib_sizes = np.unique(
            np.round(np.logspace(np.log10(min_lib * 2), np.log10(N), 5)).astype(int)
        ).tolist()
    lib_sizes = [int(L) for L in lib_sizes if 2 <= int(L) <= N]
    if len(lib_sizes) < 2:
        raise KuantValueError(
            f"kuant.ccm: not enough valid library sizes (need at least "
            f"2); got {lib_sizes}.  [KE-VAL-RANGE]"
        )

    rho_xy = np.empty(len(lib_sizes), dtype=np.float64)
    rho_yx = np.empty(len(lib_sizes), dtype=np.float64)
    for i, L in enumerate(lib_sizes):
        # Predict y from x's shadow manifold.
        scores_xy = [
            _cross_map_score(Ex, ty, L, seed=seed + s * 1000 + i) for s in range(int(n_seeds))
        ]
        # Predict x from y's shadow manifold.
        scores_yx = [
            _cross_map_score(Ey, tx, L, seed=seed + s * 1000 + i + 500000)
            for s in range(int(n_seeds))
        ]
        rho_xy[i] = float(np.nanmean(scores_xy))
        rho_yx[i] = float(np.nanmean(scores_yx))

    conv_xy = bool(rho_xy[-1] - rho_xy[0] > convergence_threshold)
    conv_yx = bool(rho_yx[-1] - rho_yx[0] > convergence_threshold)

    return CCMResult(
        lib_sizes=np.asarray(lib_sizes, dtype=np.int64),
        rho_xy=rho_xy,
        rho_yx=rho_yx,
        convergence_xy=conv_xy,
        convergence_yx=conv_yx,
        convergence_threshold=float(convergence_threshold),
        embed_dim=int(m),
        embed_tau=int(tau),
    )


__all__ = ["CCMResult", "ccm"]

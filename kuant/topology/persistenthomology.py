"""Persistent homology of a time series or point cloud.

Persistent homology (PH) tracks how topological features — connected
components (H0), loops (H1), voids (H2), … — appear and disappear as
we grow ε-balls around each point in a metric space. Each feature has
a `birth` scale (the ε where it appears) and a `death` scale (the ε
where it merges into a larger feature or fills in). Long-lived
features (large `death - birth`) are the persistent topological
structure; short-lived pairs are noise.

For a scalar time series, we build a point cloud via **time-delay
embedding** (Takens 1981):

    p_t = (x[t], x[t+τ], x[t+2τ], …, x[t+(d-1)τ])

Under mild assumptions the embedded point cloud reconstructs the
underlying attractor's topology from any single observable of the
system. This lets us compute persistence diagrams on a plain 1D
sequence.

Uses `ripser` for the Rips-complex persistence computation. Kept as a
lazy import — kuant itself has no hard `ripser` dep.

Design: docs/kernels/topology/persistenthomology.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kuant._validation import require_dep, require_positive, warn_kuant
from kuant.errors import KuantNumericWarning


@dataclass
class PersistenceDiagram:
    """Persistence diagram — birth/death pairs per homology dimension.

    Attributes
    ----------
    diagrams : dict[int, np.ndarray]
        Maps homology dimension → (n_features, 2) array of (birth, death).
        Infinite death (feature never dies inside ε_max) is encoded as
        `np.inf`. Sorted by persistence descending.
    n_points : int
        Number of points in the input point cloud.
    max_dim : int
        Highest homology dimension computed.

    Convenience
    -----------
    `d.h0`, `d.h1`, `d.h2` return the diagrams for H0/H1/H2 (or empty
    arrays if not computed).
    """

    diagrams: dict[int, np.ndarray]
    n_points: int
    max_dim: int
    embedding_dim: int | None = None
    delay: int | None = None
    extras: dict = field(default_factory=dict)

    @property
    def h0(self) -> np.ndarray:
        return self.diagrams.get(0, np.empty((0, 2)))

    @property
    def h1(self) -> np.ndarray:
        return self.diagrams.get(1, np.empty((0, 2)))

    @property
    def h2(self) -> np.ndarray:
        return self.diagrams.get(2, np.empty((0, 2)))

    def persistences(self, dim: int) -> np.ndarray:
        """Return `death - birth` per feature at the given dimension.

        Infinite-death features get `np.inf`. Sorted descending.
        """
        arr = self.diagrams.get(dim)
        if arr is None or arr.size == 0:
            return np.empty(0)
        return arr[:, 1] - arr[:, 0]

    def n_features(self, dim: int, min_persistence: float = 0.0) -> int:
        """Count features at `dim` with persistence >= `min_persistence`."""
        p = self.persistences(dim)
        if p.size == 0:
            return 0
        return int(np.sum(p >= min_persistence))


def _time_delay_embed(x: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
    """Takens time-delay embedding of a 1D series.

    Returns a `(N, embedding_dim)` array where N = len(x) - (d-1)·τ.
    """
    n = x.size
    stride = (embedding_dim - 1) * delay
    if n <= stride:
        return np.empty((0, embedding_dim), dtype=x.dtype)
    N = n - stride
    # Shape (N, d) — column j = x[j·τ : j·τ + N]
    out = np.empty((N, embedding_dim), dtype=x.dtype)
    for j in range(embedding_dim):
        out[:, j] = x[j * delay : j * delay + N]
    return out


def persistenthomology(
    series_or_cloud,
    dim: int = 1,
    embedding_dim: int = 3,
    delay: int = 1,
    max_edge_length: float | None = None,
) -> PersistenceDiagram:
    """Persistent homology of a 1D series (time-delay embedded) or point cloud.

    Parameters
    ----------
    series_or_cloud : array
        - 1D array — Takens time-delay embedded with `(embedding_dim, delay)`
          then treated as a point cloud.
        - 2D array `(n_points, n_features)` — treated directly as a
          point cloud, `embedding_dim` and `delay` ignored.
    dim : int, default 1
        Highest homology dimension to compute. Ripser convention.
    embedding_dim : int, default 3
        Only used for 1D input. Standard rule of thumb: 2·(fractal dim)+1.
    delay : int, default 1
        Delay τ for the embedding. Only used for 1D input.
    max_edge_length : float, optional
        Truncation ε for the Rips filtration. If None, ripser picks
        it (typically the enclosing radius). Set explicitly for
        reproducibility across inputs.

    Returns
    -------
    PersistenceDiagram
        With per-dimension birth/death pairs.

    Notes
    -----
    Infinite death encodes "feature persists to the end of the
    filtration". H0 always has one infinite-death class (the whole
    point set is one connected component eventually).

    Examples
    --------
    >>> import numpy as np                                                     # doctest: +SKIP
    >>> # Circle in 2D → one persistent H1 loop.                               # doctest: +SKIP
    >>> t = np.linspace(0, 2 * np.pi, 100, endpoint=False)                     # doctest: +SKIP
    >>> cloud = np.stack([np.cos(t), np.sin(t)], axis=1)                       # doctest: +SKIP
    >>> d = persistenthomology(cloud, dim=1)                                   # doctest: +SKIP
    >>> d.n_features(1, min_persistence=0.5)                                   # doctest: +SKIP
    1
    """
    try:
        from ripser import ripser
    except ImportError as e:
        require_dep(
            "ripser",
            kernel="persistenthomology",
            install="pip install ripser",
            cause=e,
        )

    require_positive(dim + 1, "dim + 1", kernel="persistenthomology", kind="int")

    arr = np.asarray(series_or_cloud, dtype=np.float64)
    if arr.ndim == 1:
        require_positive(embedding_dim, "embedding_dim", kernel="persistenthomology", kind="int")
        require_positive(delay, "delay", kernel="persistenthomology", kind="int")
        cloud = _time_delay_embed(arr, embedding_dim, delay)
        emb_d: int | None = embedding_dim
        d_delay: int | None = delay
    elif arr.ndim == 2:
        cloud = arr
        emb_d = None
        d_delay = None
    else:
        # Neither 1D nor 2D — surface a clear shape error via _validation.
        from kuant.errors import KuantShapeError

        raise KuantShapeError(
            f"kuant.persistenthomology: expected 1D series or 2D point "
            f"cloud, got shape {arr.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a 1D array (embedded via Takens) or a 2D "
            f"array of shape (n_points, n_features)"
        )

    if cloud.shape[0] < 2:
        # Persistence on <2 points is degenerate; return empty diagrams.
        return PersistenceDiagram(
            diagrams={d: np.empty((0, 2)) for d in range(dim + 1)},
            n_points=int(cloud.shape[0]),
            max_dim=dim,
            embedding_dim=emb_d,
            delay=d_delay,
        )

    # B4 — PH's asymptotics kick in around 30-50 points depending on
    # intrinsic dimension. Below ~20 the diagram is dominated by
    # boundary effects and the persistence pairs shouldn't be trusted.
    if cloud.shape[0] < 20:
        emb_desc = f"d={emb_d}, τ={d_delay}" if emb_d is not None else "point cloud"
        warn_kuant(
            kernel="persistenthomology",
            code="KW-TOPO-FEW-POINTS",
            what=(f"only {int(cloud.shape[0])} points in the cloud " f"({emb_desc})"),
            fix=(
                "results are dominated by boundary effects below ~20 points; "
                "shorten delay/embedding_dim or widen the window"
            ),
            category=KuantNumericWarning,
        )

    kwargs = {"maxdim": dim}
    if max_edge_length is not None:
        kwargs["thresh"] = float(max_edge_length)

    result = ripser(cloud, **kwargs)
    raw_diagrams = result["dgms"]  # list of (n_i, 2) arrays

    # Sort each dimension by persistence descending. Handle inf specially:
    # inf-death features go first regardless of birth.
    diagrams: dict[int, np.ndarray] = {}
    for i, dgm in enumerate(raw_diagrams):
        if dgm.size == 0:
            diagrams[i] = np.empty((0, 2))
            continue
        pers = dgm[:, 1] - dgm[:, 0]
        # np.inf sorts to the end naturally with -pers.
        order = np.argsort(-pers, kind="stable")
        diagrams[i] = dgm[order]

    return PersistenceDiagram(
        diagrams=diagrams,
        n_points=int(cloud.shape[0]),
        max_dim=dim,
        embedding_dim=emb_d,
        delay=d_delay,
    )


__all__ = ["persistenthomology", "PersistenceDiagram"]

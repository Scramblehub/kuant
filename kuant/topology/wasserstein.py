"""Distance between two persistence diagrams.

Given two persistence diagrams (birth/death arrays from
`persistenthomology`), quantify how different they are. Useful for:

- Regime-change detection: compare a rolling recent-window diagram
  against a baseline reference diagram. Spikes mark topological
  regime shifts in the underlying attractor.
- Volatility-of-topology signals: distance between successive
  overlapping diagrams tells you how quickly the phase-space shape
  is changing.

Three metrics available:

- `'wasserstein'` (default, order 2) — optimal-transport-style
  matching. Sums squared bottleneck distances over an assignment
  that minimizes the total cost. Sensitive to many small changes.
- `'bottleneck'` — the largest matched-point distance (Wasserstein
  order ∞). Sensitive only to the WORST feature displacement. More
  robust to noise but blind to small distributed shifts.
- `'sliced_wasserstein'` — much faster approximation via random 1D
  projections. Use for high-cardinality diagrams where full
  transport is expensive.

Uses `persim` for the underlying implementation. Lazy import.

Design: docs/kernels/topology/wasserstein.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import require_dep, require_positive
from kuant.errors import KuantValueError

_ALLOWED_METRICS = ("wasserstein", "bottleneck", "sliced_wasserstein")


def _to_finite_2col(dgm, name: str) -> np.ndarray:
    """Coerce a diagram-like input to a finite (n, 2) float array.

    Accepts an ndarray from `persistenthomology(...).diagrams[k]` or
    any array of `(birth, death)` rows. Rows with non-finite death
    (persistence-∞ features like the H0 total-connected component)
    are dropped — they don't participate in optimal transport.
    """
    arr = np.asarray(dgm, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise KuantValueError(
            f"kuant.wasserstein: '{name}' must be an (n, 2) array of "
            f"(birth, death) rows, got shape {arr.shape}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass e.g. `diagram.diagrams[1]` from persistenthomology, "
            f"or a numpy array of birth/death pairs"
        )
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite]


def wasserstein(diagram_a, diagram_b, metric: str = "wasserstein", n_slices: int = 50) -> float:
    """Distance between two persistence diagrams.

    Parameters
    ----------
    diagram_a, diagram_b : ndarray of shape (n, 2)
        Persistence diagrams as `(birth, death)` rows. Non-finite
        death entries are dropped before computation.
    metric : {'wasserstein', 'bottleneck', 'sliced_wasserstein'}
        Which distance to use. See module docstring for choice
        guidance.
    n_slices : int, default 50
        Random 1D projections for `'sliced_wasserstein'`. Ignored
        for the other metrics.

    Returns
    -------
    float
        Non-negative distance. Zero iff the finite parts of the
        two diagrams match exactly (both empty is also zero).

    Notes
    -----
    Diagram-diagram distances are computed after infinity-death
    filtering. This means an H0 diagram loses its one "everything is
    connected" class — the finite pairing left over is the noise
    structure, which is the informative part for regime work anyway.

    Examples
    --------
    >>> import numpy as np                                                       # doctest: +SKIP
    >>> from kuant.topology import persistenthomology, wasserstein               # doctest: +SKIP
    >>> t = np.linspace(0, 2 * np.pi, 60, endpoint=False)                        # doctest: +SKIP
    >>> a = np.stack([np.cos(t), np.sin(t)], axis=1)                             # doctest: +SKIP
    >>> b = np.stack([np.cos(t), 1.05 * np.sin(t)], axis=1)                      # slight stretch
    >>> d_a = persistenthomology(a, dim=1)                                       # doctest: +SKIP
    >>> d_b = persistenthomology(b, dim=1)                                       # doctest: +SKIP
    >>> wasserstein(d_a.h1, d_b.h1) > 0                                          # doctest: +SKIP
    True
    """
    if metric not in _ALLOWED_METRICS:
        raise KuantValueError(
            f"kuant.wasserstein: 'metric' must be one of {_ALLOWED_METRICS}, "
            f"got {metric!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_METRICS}"
        )
    require_positive(n_slices, "n_slices", kernel="wasserstein", kind="int")

    try:
        from persim import bottleneck as _bottleneck
        from persim import sliced_wasserstein as _sliced
        from persim import wasserstein as _wasserstein
    except ImportError as e:
        require_dep(
            "persim",
            kernel="wasserstein",
            install="pip install persim",
            cause=e,
        )

    a = _to_finite_2col(diagram_a, "diagram_a")
    b = _to_finite_2col(diagram_b, "diagram_b")

    # Both empty → zero distance (agreement on "no finite features").
    if a.shape[0] == 0 and b.shape[0] == 0:
        return 0.0

    if metric == "wasserstein":
        return float(_wasserstein(a, b))
    if metric == "bottleneck":
        return float(_bottleneck(a, b))
    # sliced_wasserstein
    return float(_sliced(a, b, M=int(n_slices)))


__all__ = ["wasserstein"]

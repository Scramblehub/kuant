"""Long-form → wide-panel conversion (pivot without pandas).

Given three parallel 1D arrays `(index, name, value)` — the canonical
long form for a panel — reshape into a wide `(T, N)` matrix where rows
are unique index positions and columns are unique names.

This is the one primitive everyone re-writes from pandas: `df.pivot(
index='date', columns='ticker', values='return')`. Doing it in numpy
directly avoids a pandas dep and gives you back arrays kuant's other
kernels already understand.

Missing (index, name) combinations become NaN. Duplicate (index, name)
pairs raise `KuantValueError` — the user must aggregate them upstream
(the "which return does 2024-01-02 × AAPL get?" question has no
non-arbitrary answer inside a pivot).

Design: docs/kernels/data/panelize.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_dep,
    require_equal_length,
)
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class PanelResult:
    """Wide `(T, N)` panel plus its row + column indices.

    Attributes
    ----------
    values : 2D np.ndarray, shape (T, N)
        The panel. Cells with no source row are NaN. Always float dtype
        so NaN can be represented.
    row_index : 1D np.ndarray
        The unique index positions (rows), sorted ascending.
    col_index : 1D np.ndarray
        The unique names (columns), sorted ascending.
    n_source_rows : int
        How many input rows were consumed. `T * N - count_of_NaN` — a
        density check.
    """

    values: np.ndarray
    row_index: np.ndarray
    col_index: np.ndarray
    n_source_rows: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def summary(self) -> str:
        n_cells = int(self.values.size)
        n_finite = int(np.isfinite(self.values).sum())
        density = 100.0 * n_finite / n_cells if n_cells else 0.0
        parts = [
            "=== PanelResult ===",
            f"shape:             {self.values.shape[0]} × {self.values.shape[1]}",
            f"source rows:       {self.n_source_rows}",
            f"density:           {density:.1f}%  ({n_finite:,} finite / {n_cells:,} cells)",
        ]
        return "\n".join(parts)

    def to_parquet(self, path) -> None:
        """Write the panel to a parquet file.

        Columns: `row_index` plus one column per name (as string).
        Requires `pyarrow`.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="panelize.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        cols = {"row_index": pa.array(self.row_index)}
        for j, name in enumerate(self.col_index):
            cols[str(name)] = pa.array(self.values[:, j])
        pq.write_table(pa.table(cols), path)


def panelize(index, name, value) -> PanelResult:
    """Pivot three parallel 1D arrays into a wide `(T, N)` panel.

    Parameters
    ----------
    index : 1D array of length K
        Row identifier per source row (typically date, timestamp, or
        integer position). Duplicates within a name are rejected.
    name : 1D array of length K
        Column identifier per source row (typically ticker or asset id).
        Any hashable dtype (string, int, datetime) is accepted.
    value : 1D array of length K
        Per-row measurement. Coerced to float64.

    Returns
    -------
    PanelResult
        `.values`      (T, N) float array. NaN where no source row.
        `.row_index`   1D sorted ascending, length T.
        `.col_index`   1D sorted ascending, length N.
        `.n_source_rows` the input length K.

    Notes
    -----
    - Duplicate `(index, name)` pairs raise `KuantValueError`. Aggregate
      duplicates upstream (e.g. sum, last-wins) before calling.
    - The output is dense — `T * N` cells whether populated or not. For
      very jagged panels (many names with narrow coverage), consider a
      sparse alternative outside kuant.
    - Empty inputs produce an `(0, 0)` panel without error.

    Examples
    --------
    >>> import numpy as np
    >>> # Two tickers, three dates, one missing observation.
    >>> idx = np.array([1, 1, 2, 2, 3])
    >>> nm  = np.array(['A', 'B', 'A', 'B', 'A'])
    >>> val = np.array([10.0, 100, 20, 200, 30])
    >>> r = panelize(idx, nm, val)
    >>> r.shape
    (3, 2)
    >>> r.row_index.tolist()
    [1, 2, 3]
    >>> r.col_index.tolist()
    ['A', 'B']
    >>> r.values.tolist()   # (3, B) is missing → NaN
    [[10.0, 100.0], [20.0, 200.0], [30.0, nan]]
    """
    idx = np.asarray(index)
    nm = np.asarray(name)
    val = np.asarray(value, dtype=np.float64)

    require_1d(idx, "index", kernel="panelize")
    require_1d(nm, "name", kernel="panelize")
    require_1d(val, "value", kernel="panelize")
    require_equal_length(idx, "index", nm, "name", kernel="panelize")
    require_equal_length(idx, "index", val, "value", kernel="panelize")

    if idx.size == 0:
        return PanelResult(
            values=np.empty((0, 0), dtype=np.float64),
            row_index=np.empty(0, dtype=idx.dtype),
            col_index=np.empty(0, dtype=nm.dtype),
            n_source_rows=0,
        )

    # Unique row + column labels, sorted ascending; also positions.
    row_index, row_pos = np.unique(idx, return_inverse=True)
    col_index, col_pos = np.unique(nm, return_inverse=True)

    T, N = row_index.size, col_index.size

    # Duplicate (index, name) detection via composite id = row_pos * N + col_pos.
    # Duplicates would produce collisions in this flat index.
    flat = row_pos.astype(np.int64) * N + col_pos.astype(np.int64)
    if flat.size > 1:
        sorted_flat = np.sort(flat)
        dup_mask = sorted_flat[1:] == sorted_flat[:-1]
        if bool(dup_mask.any()):
            dup_flat = int(sorted_flat[np.where(dup_mask)[0][0]])
            dup_row = dup_flat // N
            dup_col = dup_flat % N
            raise KuantValueError(
                f"kuant.panelize: duplicate (index, name) pair "
                f"({row_index[dup_row]!r}, {col_index[dup_col]!r}) "
                f"appears more than once.  [KE-VAL-DUPLICATE]\n"
                f"  → Fix: aggregate duplicates upstream (sum, mean, or "
                f"last-wins) before calling — one row per "
                f"(index, name) pair"
            )

    values = np.full((T, N), np.nan, dtype=np.float64)
    values[row_pos, col_pos] = val
    return PanelResult(
        values=values,
        row_index=row_index,
        col_index=col_index,
        n_source_rows=int(idx.size),
    )


def unpanelize(panel: PanelResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of `panelize`: wide `(T, N)` → three long-form arrays.

    Emits one row per FINITE cell (NaN cells are dropped). Order is
    row-major on the panel's `.row_index` × `.col_index`.

    Returns
    -------
    (index, name, value) : three 1D arrays of equal length
        Each cell `(t, n)` where `panel.values[t, n]` is finite becomes
        the triple `(panel.row_index[t], panel.col_index[n],
        panel.values[t, n])`.

    Examples
    --------
    >>> import numpy as np
    >>> idx = np.array([1, 1, 2])
    >>> nm = np.array(['A', 'B', 'A'])
    >>> val = np.array([10.0, 100, 20])
    >>> p = panelize(idx, nm, val)
    >>> idx2, nm2, val2 = unpanelize(p)
    >>> len(idx2)   # (2, B) was NaN → dropped
    3
    """
    if not isinstance(panel, PanelResult):
        raise KuantShapeError(
            f"kuant.unpanelize: expected a PanelResult, got "
            f"{type(panel).__name__}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass the result of a `panelize(...)` call"
        )
    T, N = panel.values.shape
    if T == 0 or N == 0:
        return (
            np.empty(0, dtype=panel.row_index.dtype),
            np.empty(0, dtype=panel.col_index.dtype),
            np.empty(0, dtype=np.float64),
        )
    mask = np.isfinite(panel.values)
    # Row-major cell coordinates.
    row_ids, col_ids = np.where(mask)
    return (
        panel.row_index[row_ids],
        panel.col_index[col_ids],
        panel.values[row_ids, col_ids],
    )


__all__ = ["panelize", "unpanelize", "PanelResult"]

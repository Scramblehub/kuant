"""Merge multiple partial-coverage panels into one wider panel.

When vendor A covers ticker+date grid (T_A, N_A) and vendor B covers
(T_B, N_B) with partial overlap, `stitch` unions the row and column
indices and produces a single `(T_merged, N_merged)` panel under a
chosen conflict-resolution rule.

Two rules:

- **`'first_wins'`** — the first panel's value wins in any overlapping
  cell. Later panels only fill cells the earlier ones left NaN.
  Suitable for "prefer my clean vendor over the noisy backup vendor".
- **`'last_wins'`** — the last panel's value overrides. Useful when
  each successive panel is a correction to the prior one.

Design: docs/kernels/data/stitch.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import require_range, warn_kuant
from kuant.errors import KuantNumericWarning, KuantValueError

from .panelize import PanelResult

_ALLOWED_METHODS = ("first_wins", "last_wins")


def stitch(*panels, method: str = "first_wins") -> PanelResult:
    """Merge two or more `PanelResult`s into a single panel.

    Parameters
    ----------
    *panels : PanelResult
        Two or more panels. All must be `PanelResult` instances (from
        `panelize`) so we can trust their row/col index semantics.
    method : {'first_wins', 'last_wins'}
        Conflict resolution when the same `(row, col)` cell has a
        finite value in more than one input panel.

    Returns
    -------
    PanelResult
        The merged panel. `.n_source_rows` sums across inputs.

    Notes
    -----
    - Row and column indices are unioned and sorted ascending.
    - Row and column dtypes must be compatible across all inputs
      (numeric with numeric, datetime with datetime). Mixing raises
      `KuantValueError`.
    - Cells where multiple panels supply finite values but disagree by
      more than 1e-9 fire a `KuantNumericWarning` — that's usually a
      vendor-adjustment mismatch worth flagging.
    """
    if method not in _ALLOWED_METHODS:
        raise KuantValueError(
            f"kuant.stitch: 'method' must be one of {_ALLOWED_METHODS}, "
            f"got {method!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_METHODS}"
        )
    require_range(
        len(panels),
        "number of panels",
        kernel="stitch",
        lo=2,
        hi=float("inf"),
    )
    for i, p in enumerate(panels):
        if not isinstance(p, PanelResult):
            raise KuantValueError(
                f"kuant.stitch: panels[{i}] is not a PanelResult (got "
                f"{type(p).__name__}).  [KE-SHAPE-EXPECTED]\n"
                f"  → Fix: pass results of `panelize(...)` — plain "
                f"ndarrays are not supported"
            )

    # Compatible dtypes across all inputs.
    row_kinds = {p.row_index.dtype.kind for p in panels}
    col_kinds = {p.col_index.dtype.kind for p in panels}
    numeric = {"i", "u", "f"}
    if row_kinds & numeric and (row_kinds - numeric):
        raise KuantValueError(
            f"kuant.stitch: row-index dtype kinds {sorted(row_kinds)} "
            f"cannot be safely unioned; refusing to coerce.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: cast every panel's row_index to a common dtype "
            f"before calling"
        )
    if col_kinds & numeric and (col_kinds - numeric):
        raise KuantValueError(
            f"kuant.stitch: col-index dtype kinds {sorted(col_kinds)} "
            f"cannot be safely unioned; refusing to coerce.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: cast every panel's col_index to a common dtype"
        )

    # Union row + column indices.
    all_rows = panels[0].row_index
    for p in panels[1:]:
        all_rows = np.union1d(all_rows, p.row_index)
    all_cols = panels[0].col_index
    for p in panels[1:]:
        all_cols = np.union1d(all_cols, p.col_index)

    T, N = all_rows.size, all_cols.size
    merged = np.full((T, N), np.nan, dtype=np.float64)

    # Track conflicts for the disagreement warning.
    n_conflicts = 0
    conflict_first = None  # (row_val, col_val, v_a, v_b)

    # Loop panels; each supplies (row_positions_in_merged, col_positions_in_merged).
    for panel_i, p in enumerate(panels):
        # Positions of this panel's rows in the merged row index.
        row_pos = np.searchsorted(all_rows, p.row_index)
        col_pos = np.searchsorted(all_cols, p.col_index)
        # Build (T_i, N_i) → (T, N) scatter target block.
        # Efficient path: expand to the target shape via broadcasting.
        # Use nested indexing rather than a full O(T*N) allocation.
        source = p.values
        finite = np.isfinite(source)

        # Only touch cells where source has finite values.
        for i, r in enumerate(row_pos):
            for j, c in enumerate(col_pos):
                if not finite[i, j]:
                    continue
                new_val = source[i, j]
                cur_val = merged[r, c]
                if np.isfinite(cur_val):
                    # Overlap — check for disagreement.
                    if abs(cur_val - new_val) > 1e-9:
                        n_conflicts += 1
                        if conflict_first is None:
                            conflict_first = (
                                all_rows[r],
                                all_cols[c],
                                float(cur_val),
                                float(new_val),
                            )
                    if method == "last_wins":
                        merged[r, c] = new_val
                    # else first_wins: skip (keep cur_val)
                else:
                    merged[r, c] = new_val

    if n_conflicts > 0:
        r, c, va, vb = conflict_first
        warn_kuant(
            kernel="stitch",
            code="KW-STITCH-DISAGREE",
            what=(
                f"{n_conflicts} cell(s) had finite disagreement across "
                f"panels; first at ({r!r}, {c!r}): {va:.6g} vs {vb:.6g}"
            ),
            fix=(
                "if the vendors' adjustment conventions differ, adjust "
                "one to match the other before stitching, or accept the "
                "first_wins / last_wins policy you chose"
            ),
            category=KuantNumericWarning,
        )

    return PanelResult(
        values=merged,
        row_index=all_rows,
        col_index=all_cols,
        n_source_rows=sum(int(p.n_source_rows) for p in panels),
    )


__all__ = ["stitch"]

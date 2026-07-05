"""Per-asset and per-group P&L contribution.

Given positions and returns aligned on the same `(T, N)` grid, the
per-cell P&L is:

    pnl[t, i] = positions[t, i] * returns[t, i]

Total portfolio P&L at each bar is the sum across names. Per-name
contribution is the sum across time. Optionally aggregate by group
labels for sector, factor, or bucket attribution.

Design: docs/kernels/portfolio/contribution.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kuant._validation import (
    require_1d,
    require_2d,
    require_dep,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantShapeError


@dataclass
class ContributionResult:
    """Per-asset and (optionally) per-group P&L attribution.

    Attributes
    ----------
    per_bar_pnl : 2D np.ndarray, shape (T, N)
        Element-wise `positions * returns`.
    total_by_asset : 1D np.ndarray, length N
        Sum across time for each name.
    total_by_bar : 1D np.ndarray, length T
        Sum across names at each bar.
    total : float
        Grand total. Sum of `total_by_asset` (equivalently
        `total_by_bar`).
    per_group : dict[str, float] or None
        If group labels were passed, per-group aggregated P&L; None
        otherwise.
    n_positions : int
        Total count of finite (positions * returns) products used.
    """

    per_bar_pnl: np.ndarray
    total_by_asset: np.ndarray
    total_by_bar: np.ndarray
    total: float
    per_group: dict | None
    n_positions: int
    asset_names: np.ndarray | None = field(default=None)

    def summary(self) -> str:
        parts = [
            "=== ContributionResult ===",
            f"total P&L:          {self.total:+.6f}",
            f"shape (T, N):       {self.per_bar_pnl.shape}",
            f"n_positions:        {self.n_positions}",
        ]
        # Show top 5 contributors by absolute value.
        top = np.argsort(-np.abs(self.total_by_asset))[:5]
        parts.append("")
        parts.append("top 5 contributors by |P&L|:")
        for j in top:
            name = str(self.asset_names[j]) if self.asset_names is not None else f"asset{int(j)}"
            parts.append(f"  {name:<20s} {self.total_by_asset[j]:+.6f}")
        if self.per_group is not None:
            parts.append("")
            parts.append("per-group P&L:")
            for g, v in sorted(self.per_group.items(), key=lambda kv: -abs(kv[1])):
                parts.append(f"  {str(g):<20s} {v:+.6f}")
        return "\n".join(parts)

    def to_parquet(self, path) -> None:
        """Write per-asset totals to parquet. Requires pyarrow.

        Columns: asset_name (or 'asset<index>'), total_pnl.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="contribution.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        if self.asset_names is not None:
            names = [str(x) for x in self.asset_names]
        else:
            names = [f"asset{i}" for i in range(self.total_by_asset.size)]
        table = pa.table(
            {
                "asset": pa.array(names),
                "total_pnl": pa.array(self.total_by_asset),
            }
        )
        pq.write_table(table, path)


def contribution(
    positions,
    returns,
    group=None,
    asset_names=None,
) -> ContributionResult:
    """Per-asset (and optional per-group) P&L attribution.

    Parameters
    ----------
    positions : 2D array, shape (T, N)
        Position size per bar and name. Units are up to you: notional,
        weight, shares. The P&L is `positions * returns`.
    returns : 2D array, shape (T, N)
        Periodic returns aligned to `positions`.
    group : 1D array of length N, optional
        Group label per asset. If supplied, `.per_group` in the result
        aggregates P&L by unique label.
    asset_names : 1D array of length N, optional
        Names for each column, used in summary and to_parquet.

    Returns
    -------
    ContributionResult

    Warnings
    --------
    `KuantNumericWarning` (`KW-CONTRIB-PARTIAL-COVERAGE`) if fewer
    than 80% of the `(T, N)` cells are finite. Partial coverage is
    the pattern where a wrong join or missing-data-window produces
    silently-underrepresented names.

    Notes
    -----
    - NaN cells are treated as zero P&L. That matches "no position or
      no return at this bar means no contribution".
    - `positions` and `returns` must share the same `(T, N)` shape.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[1.0, 2, 0], [1, 2, 1]])
    >>> returns   = np.array([[0.01, 0.02, 0.03], [0.02, -0.01, 0.05]])
    >>> r = contribution(positions, returns, asset_names=["A", "B", "C"])
    >>> r.total_by_asset.tolist()
    [0.03, 0.02, 0.05]
    """
    pos = np.asarray(positions, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    require_2d(pos, "positions", kernel="contribution")
    require_2d(ret, "returns", kernel="contribution")
    if pos.shape != ret.shape:
        raise KuantShapeError(
            f"kuant.contribution: 'positions' and 'returns' must share "
            f"shape, got {pos.shape} vs {ret.shape}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: align them on a common (T, N) grid before "
            f"calling (see kuant.data.align + panelize)"
        )
    T, N = pos.shape

    per_bar = pos * ret
    # NaN → 0 for the totals; keep NaN in the raw per_bar view.
    per_bar_zero = np.nan_to_num(per_bar, nan=0.0)
    total_by_asset = per_bar_zero.sum(axis=0)
    total_by_bar = per_bar_zero.sum(axis=1)
    total = float(total_by_asset.sum())

    n_positions = int(np.isfinite(per_bar).sum())
    coverage = n_positions / (T * N) if T * N else 0.0
    if coverage < 0.8 and T * N > 0:
        warn_kuant(
            kernel="contribution",
            code="KW-CONTRIB-PARTIAL-COVERAGE",
            what=(
                f"only {n_positions}/{T * N} cells ({100 * coverage:.1f}%) "
                f"have finite (position, return) pairs"
            ),
            fix=(
                "positions and returns don't fully overlap; check the "
                "join / panelization, or accept that under-covered names "
                "will be underrepresented in the P&L attribution"
            ),
            category=KuantNumericWarning,
        )

    per_group = None
    if group is not None:
        group_arr = np.asarray(group)
        require_1d(group_arr, "group", kernel="contribution")
        if group_arr.size != N:
            raise KuantShapeError(
                f"kuant.contribution: 'group' length {group_arr.size} "
                f"does not match number of columns {N}.  "
                f"[KE-SHAPE-EQUAL-LEN]\n"
                f"  → Fix: pass one group label per column"
            )
        per_group = {}
        for lbl in np.unique(group_arr):
            mask = group_arr == lbl
            per_group[str(lbl)] = float(total_by_asset[mask].sum())

    asset_names_arr = None
    if asset_names is not None:
        names_arr = np.asarray(asset_names)
        if names_arr.size != N:
            raise KuantShapeError(
                f"kuant.contribution: 'asset_names' length {names_arr.size} "
                f"does not match number of columns {N}.  "
                f"[KE-SHAPE-EQUAL-LEN]\n"
                f"  → Fix: pass one name per column"
            )
        asset_names_arr = names_arr

    return ContributionResult(
        per_bar_pnl=per_bar,
        total_by_asset=total_by_asset,
        total_by_bar=total_by_bar,
        total=total,
        per_group=per_group,
        n_positions=n_positions,
        asset_names=asset_names_arr,
    )


__all__ = ["contribution", "ContributionResult"]

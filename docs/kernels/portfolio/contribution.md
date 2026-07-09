# contribution: Per-asset and per-group P&L attribution

## Purpose

Given aligned `(T, N)` panels of positions and returns, compute:

- Per-bar per-asset P&L: `positions * returns` element-wise.
- Totals across time (per asset), across assets (per bar), and the
  grand total.
- Optional per-group aggregation (sector, factor, bucket, ...).

The canonical "who made or lost money and when" attribution. Used to
diagnose whether portfolio P&L is concentrated in a small number of
names or diffused across the book.

## Public API

```python
from kuant.portfolio import contribution

r = contribution(positions, returns)
r = contribution(positions, returns,
                 group=sector_labels,
                 asset_names=tickers)
r.per_bar_pnl        # (T, N)
r.total_by_asset     # (N,)
r.total_by_bar       # (T,)
r.total              # scalar
r.per_group          # dict[str, float] or None
r.n_positions        # count of finite (pos, ret) pairs
print(r.summary())
r.to_parquet("attribution.parquet")
```

- `positions` — 2D (T, N). Units are user-defined: notional, weight,
  shares. The kernel does not care as long as `positions * returns`
  has the meaning of P&L.
- `returns` — 2D (T, N), same shape.
- `group` — optional (N,) labels. When supplied, `per_group`
  aggregates `total_by_asset` by unique label.
- `asset_names` — optional (N,) names. Used in `.summary()` and
  written as the asset column of `.to_parquet()`.

## Design decisions

### 1. NaN-as-zero for totals

`np.nan_to_num(per_bar, nan=0.0)` before summing. "Missing position
or missing return means no contribution" is the correct behavior for
attribution: a NaN cell should not poison the entire column total.
The raw `per_bar_pnl` keeps NaNs so downstream inspection can still
distinguish "zero P&L" from "no data".

### 2. `KW-CONTRIB-PARTIAL-COVERAGE` at 80% threshold

Emit a `KuantNumericWarning` when fewer than 80% of the `T * N`
cells are finite. The pattern this catches is a bad join or
missing-data window that silently drops rows and understates a name
that was actually in the book. The threshold is high enough that
routine short-lived listings do not trigger, low enough that a wrong
merge does.

### 3. Grand total from `total_by_asset`

`total = total_by_asset.sum()`. Mathematically identical to
`total_by_bar.sum()`; using the asset-level totals matches the way
the per-group aggregation is computed and keeps one canonical
reduction path.

### 4. Optional per-group aggregation

`per_group` maps each unique label in `group` to the sum of
`total_by_asset` over the matching mask. Labels are stored as
strings for parquet friendliness. `None` when no group vector is
supplied.

### 5. `to_parquet` writes per-asset totals only

The full `(T, N)` panel is expensive to serialize and rarely needed
downstream. `to_parquet` writes columns `asset, total_pnl` and lets
the caller reconstruct the panel from the raw `per_bar_pnl` array
if required.

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| `positions.shape != returns.shape` | `KuantShapeError [KE-SHAPE-EXPECTED]` with align hint |
| `positions.ndim != 2` or `returns.ndim != 2` | raised by `require_2d` |
| `group.size != N` | `KuantShapeError [KE-SHAPE-EQUAL-LEN]` |
| `asset_names.size != N` | `KuantShapeError [KE-SHAPE-EQUAL-LEN]` |
| Coverage < 80% | `KuantNumericWarning [KW-CONTRIB-PARTIAL-COVERAGE]` |
| All-NaN inputs | totals are 0.0, warning fires |
| `pyarrow` missing at `to_parquet` | raises via `require_dep` |

## Cross-check tests

- Golden hand-computed 2-bar 3-asset panel.
- Group-aggregation round-trip.
- Partial-coverage warning fires at scattered NaN rate.
- Parquet round-trip through pyarrow.

`tests/portfolio/test_contribution.py`.

## References

- Standard performance-attribution decomposition. No specific
  literature citation.

## Related kernels

- `kuant.portfolio.drawdown` — feed `total_by_bar` cumulated into an
  equity curve to check the drawdown of individual attribution
  slices.
- `kuant.portfolio.sharperatio`, `kuant.portfolio.sortinoratio` — run
  on `total_by_bar` for whole-book risk-adjusted return.
- `kuant.data.align` and `panelize` — build the aligned `(T, N)`
  panels this kernel expects.

"""Tests for kuant.data.panelize + unpanelize."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.data.panelize import PanelResult, panelize, unpanelize
from kuant.errors import KuantShapeError, KuantValueError


# ---------- basic pivot ---------------------------------------------------


def test_two_tickers_three_dates_missing_cell():
    idx = np.array([1, 1, 2, 2, 3])
    nm = np.array(["A", "B", "A", "B", "A"])
    val = np.array([10.0, 100, 20, 200, 30])
    r = panelize(idx, nm, val)
    assert r.shape == (3, 2)
    assert r.row_index.tolist() == [1, 2, 3]
    assert r.col_index.tolist() == ["A", "B"]
    assert r.values[0].tolist() == [10.0, 100.0]
    assert r.values[1].tolist() == [20.0, 200.0]
    assert r.values[2][0] == 30.0
    assert np.isnan(r.values[2][1])


def test_returns_panel_result_type():
    idx = np.array([1])
    r = panelize(idx, np.array(["A"]), np.array([1.0]))
    assert isinstance(r, PanelResult)


def test_row_and_col_index_sorted_ascending():
    """Input in any order; row and col indices come back sorted."""
    idx = np.array([3, 1, 2])
    nm = np.array(["Z", "A", "M"])
    val = np.array([30.0, 10, 20])
    r = panelize(idx, nm, val)
    assert r.row_index.tolist() == [1, 2, 3]
    assert r.col_index.tolist() == ["A", "M", "Z"]
    # Value at (1, A) is 10, (2, M) is 20, (3, Z) is 30.
    assert r.values[0, 0] == 10.0
    assert r.values[1, 1] == 20.0
    assert r.values[2, 2] == 30.0


def test_empty_input_returns_empty_panel():
    r = panelize(np.array([]), np.array([]), np.array([]))
    assert r.shape == (0, 0)
    assert r.n_source_rows == 0


def test_single_row_single_column():
    r = panelize(np.array([1]), np.array(["A"]), np.array([42.0]))
    assert r.shape == (1, 1)
    assert r.values[0, 0] == 42.0


def test_int_names():
    """Names can be int, not just strings."""
    idx = np.array([1, 2])
    nm = np.array([100, 200])
    val = np.array([1.0, 2.0])
    r = panelize(idx, nm, val)
    assert r.col_index.tolist() == [100, 200]


def test_datetime_index():
    idx = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    nm = np.array(["A", "A"])
    val = np.array([10.0, 20.0])
    r = panelize(idx, nm, val)
    assert r.row_index.dtype.kind == "M"


# ---------- shape and density -------------------------------------------


def test_density_reflects_missing_cells():
    """A 3x2 panel with one missing cell has 5/6 density."""
    idx = np.array([1, 1, 2, 2, 3])
    nm = np.array(["A", "B", "A", "B", "A"])
    val = np.array([10.0, 100, 20, 200, 30])
    r = panelize(idx, nm, val)
    n_finite = int(np.isfinite(r.values).sum())
    assert n_finite == 5
    assert r.values.size == 6


def test_shape_property_matches_values():
    r = panelize(
        np.array([1, 1, 2]),
        np.array(["A", "B", "A"]),
        np.array([1.0, 2, 3]),
    )
    assert r.shape == r.values.shape


# ---------- unpanelize (inverse) ----------------------------------------


def test_unpanelize_round_trip_dense():
    """Fully-dense panel round-trips exactly."""
    idx = np.array([1, 1, 2, 2])
    nm = np.array(["A", "B", "A", "B"])
    val = np.array([10.0, 100, 20, 200])
    p = panelize(idx, nm, val)
    idx2, nm2, val2 = unpanelize(p)

    # Sort both by (idx, nm) to compare — order isn't guaranteed identical.
    def _key(a, b, c):
        return sorted(zip(a.tolist(), b.tolist(), c.tolist()))

    assert _key(idx, nm, val) == _key(idx2, nm2, val2)


def test_unpanelize_drops_nan_cells():
    """Sparse panel: NaN cells don't appear in the long-form output."""
    idx = np.array([1, 1, 2])
    nm = np.array(["A", "B", "A"])
    val = np.array([10.0, 100, 20])
    p = panelize(idx, nm, val)
    idx2, _, _ = unpanelize(p)
    assert len(idx2) == 3  # Not 4 — (2, B) was NaN and dropped.


def test_unpanelize_empty_panel():
    empty = panelize(np.array([]), np.array([]), np.array([]))
    idx2, nm2, val2 = unpanelize(empty)
    assert idx2.size == nm2.size == val2.size == 0


# ---------- error contract -----------------------------------------------


def test_reject_duplicate_pair():
    """Same (index, name) twice → error with first duplicate named."""
    idx = np.array([1, 1, 2])
    nm = np.array(["A", "A", "B"])  # (1, A) appears twice
    val = np.array([10.0, 11, 20])
    with pytest.raises(KuantValueError) as exc:
        panelize(idx, nm, val)
    m = str(exc.value)
    assert "duplicate" in m
    assert "'A'" in m or "A" in m


def test_reject_length_mismatch():
    with pytest.raises(Exception) as exc:  # KuantShapeError via require_equal_length
        panelize(np.array([1, 2]), np.array(["A"]), np.array([1.0]))
    assert "length" in str(exc.value) or "match" in str(exc.value)


def test_reject_2d_input():
    with pytest.raises(Exception):
        panelize(
            np.zeros((3, 2), dtype=np.int64),
            np.array(["A", "A", "A"]),
            np.array([1.0, 2, 3]),
        )


def test_unpanelize_rejects_non_panel_result():
    with pytest.raises(KuantShapeError):
        unpanelize(np.zeros((3, 2)))  # not a PanelResult


# ---------- summary + parquet -------------------------------------------


def test_summary_reports_density():
    r = panelize(
        np.array([1, 1, 2]),
        np.array(["A", "B", "A"]),
        np.array([1.0, 2, 3]),
    )
    s = r.summary()
    assert "PanelResult" in s
    assert "density" in s


def test_to_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    idx = np.array([1, 2, 3])
    nm = np.array(["A", "B", "A"])
    val = np.array([10.0, 100, 20])
    r = panelize(idx, nm, val)

    path = tmp_path / "panel.parquet"
    r.to_parquet(path)
    table = pq.read_table(path)
    cols = set(table.column_names)
    assert cols == {"row_index", "A", "B"}
    assert table.num_rows == r.shape[0]


# ---------- integration: composes with align ----------------------------


def test_panel_from_two_ticker_series():
    """Realistic use: build a panel from two per-ticker series with
    different date coverage, then verify sparsity."""
    dates_a = np.array([1, 2, 3, 4])
    dates_b = np.array([3, 4, 5, 6])
    idx = np.concatenate([dates_a, dates_b])
    nm = np.concatenate([np.full(4, "A"), np.full(4, "B")])
    val = np.concatenate([np.arange(4.0), np.arange(4.0) * 10])
    r = panelize(idx, nm, val)
    # Union of dates is [1..6] → 6 rows; two tickers → 2 cols.
    assert r.shape == (6, 2)
    # A is finite on [1..4], NaN on [5, 6].
    assert np.isnan(r.values[4:, 0]).all()  # A is column 0 by sort order
    # B is NaN on [1, 2], finite on [3..6].
    assert np.isnan(r.values[:2, 1]).all()

"""Tests for kuant.data.stitch."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.data.panelize import PanelResult, panelize
from kuant.data.stitch import stitch
from kuant.errors import KuantNumericWarning, KuantValueError


# ---------- basic merge ---------------------------------------------------


def _panel_from(idx, name, val):
    return panelize(
        np.asarray(idx),
        np.asarray(name),
        np.asarray(val, dtype=np.float64),
    )


def test_disjoint_panels_union_both_axes():
    """Panel A covers (dates 1..2, name A), B covers (dates 3..4, name B)."""
    a = _panel_from([1, 2], ["A", "A"], [10, 20])
    b = _panel_from([3, 4], ["B", "B"], [300, 400])
    r = stitch(a, b)
    assert r.shape == (4, 2)
    assert r.row_index.tolist() == [1, 2, 3, 4]
    assert r.col_index.tolist() == ["A", "B"]
    # A column has values at rows 1, 2 only.
    assert r.values[:2, 0].tolist() == [10.0, 20.0]
    assert np.isnan(r.values[2:, 0]).all()
    # B column has values at rows 3, 4 only.
    assert np.isnan(r.values[:2, 1]).all()
    assert r.values[2:, 1].tolist() == [300.0, 400.0]


def test_first_wins_prefers_first_panel_in_overlap():
    """A and B both have (1, X); first_wins keeps A's value."""
    a = _panel_from([1], ["X"], [10.0])
    b = _panel_from([1], ["X"], [999.0])
    r = stitch(a, b, method="first_wins")
    assert r.values[0, 0] == 10.0


def test_last_wins_prefers_later_panel_in_overlap():
    a = _panel_from([1], ["X"], [10.0])
    b = _panel_from([1], ["X"], [999.0])
    r = stitch(a, b, method="last_wins")
    assert r.values[0, 0] == 999.0


def test_second_panel_fills_first_panel_nan_gaps():
    """A covers (1, X); B covers (1, Y). Union has both filled."""
    a = _panel_from([1, 2], ["X", "X"], [10, 20])
    b = _panel_from([1, 2], ["Y", "Y"], [100, 200])
    r = stitch(a, b)
    assert r.shape == (2, 2)
    # No NaN in the merged panel — the two panels' coverage together is complete.
    assert np.isnan(r.values).sum() == 0


def test_three_panel_union():
    a = _panel_from([1], ["A"], [1.0])
    b = _panel_from([2], ["B"], [2.0])
    c = _panel_from([3], ["C"], [3.0])
    r = stitch(a, b, c)
    assert r.shape == (3, 3)
    # Diagonal has the values.
    assert r.values[0, 0] == 1.0
    assert r.values[1, 1] == 2.0
    assert r.values[2, 2] == 3.0


def test_n_source_rows_sums_across_inputs():
    a = _panel_from([1, 2], ["A", "A"], [10, 20])
    b = _panel_from([1, 2, 3], ["B", "B", "B"], [100, 200, 300])
    r = stitch(a, b)
    assert r.n_source_rows == 2 + 3


# ---------- disagreement warning -----------------------------------------


def test_disagreement_between_panels_warns():
    """Same cell with different finite values → warning."""
    a = _panel_from([1], ["X"], [10.0])
    b = _panel_from([1], ["X"], [11.0])
    with pytest.warns(KuantNumericWarning) as record:
        stitch(a, b)
    assert any("KW-STITCH-DISAGREE" in str(w.message) for w in record)


def test_agreement_no_warning():
    """Same cell with same value → no warning."""
    a = _panel_from([1], ["X"], [10.0])
    b = _panel_from([1], ["X"], [10.0])
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", KuantNumericWarning)
        stitch(a, b)  # would raise if a warning fires


def test_disagreement_first_wins_still_warns():
    """The warning fires regardless of the resolution policy."""
    a = _panel_from([1], ["X"], [10.0])
    b = _panel_from([1], ["X"], [11.0])
    with pytest.warns(KuantNumericWarning):
        stitch(a, b, method="first_wins")


# ---------- returned object contract -------------------------------------


def test_returns_panel_result():
    a = _panel_from([1], ["A"], [1.0])
    b = _panel_from([2], ["B"], [2.0])
    r = stitch(a, b)
    assert isinstance(r, PanelResult)


def test_to_parquet_roundtrip(tmp_path):
    """The stitched panel supports the PanelResult.to_parquet round-trip."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    a = _panel_from([1, 2], ["A", "A"], [10, 20])
    b = _panel_from([2, 3], ["B", "B"], [200, 300])
    r = stitch(a, b)
    path = tmp_path / "stitched.parquet"
    r.to_parquet(path)
    cols = set(pq.read_table(path).column_names)
    assert cols == {"row_index", "A", "B"}


# ---------- error contract -----------------------------------------------


def test_reject_single_panel():
    a = _panel_from([1], ["A"], [1.0])
    with pytest.raises(KuantValueError):
        stitch(a)


def test_reject_bad_method():
    a = _panel_from([1], ["A"], [1.0])
    b = _panel_from([2], ["B"], [2.0])
    with pytest.raises(KuantValueError):
        stitch(a, b, method="tie_breaker")


def test_reject_non_panel_input():
    a = _panel_from([1], ["A"], [1.0])
    with pytest.raises(KuantValueError):
        stitch(a, np.zeros((3, 2)))


def test_reject_mixed_row_dtype_kinds():
    """A numeric row_index + a datetime row_index → refuse to merge."""
    a = _panel_from([1, 2], ["A", "A"], [1.0, 2.0])
    dt_idx = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    b = _panel_from(dt_idx, np.array(["B", "B"]), np.array([1.0, 2.0]))
    with pytest.raises(KuantValueError):
        stitch(a, b)


# ---------- integration: realistic vendor merge --------------------------


def test_two_vendor_partial_overlap():
    """Realistic pattern: vendor A covers 2020-2022 for tickers {AAPL, MSFT},
    vendor B covers 2021-2023 for {MSFT, GOOG}. Merge gives 2020-2023 x 3
    tickers with overlaps resolved by first_wins (i.e., trust vendor A)."""
    a = _panel_from(
        [2020, 2020, 2021, 2021, 2022, 2022],
        ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        [100.0, 200, 110, 210, 120, 220],
    )
    b = _panel_from(
        [2021, 2021, 2022, 2022, 2023, 2023],
        ["MSFT", "GOOG", "MSFT", "GOOG", "MSFT", "GOOG"],
        [210.0, 500, 220, 510, 230, 520],
    )
    r = stitch(a, b, method="first_wins")
    assert r.shape == (4, 3)  # 2020..2023 × {AAPL, GOOG, MSFT}
    # AAPL missing in 2023.
    aapl_col = int(np.where(r.col_index == "AAPL")[0][0])
    assert np.isnan(r.values[-1, aapl_col])
    # MSFT covered by both; first-wins → vendor A's value (they agreed on 210
    # and 220 so no conflict).
    msft_col = int(np.where(r.col_index == "MSFT")[0][0])
    assert r.values[1, msft_col] == 210.0
    assert r.values[2, msft_col] == 220.0
    # 2023 MSFT comes from vendor B.
    assert r.values[3, msft_col] == 230.0

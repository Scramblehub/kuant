"""Tests for kuant.lifecycle.security."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.lifecycle.security import (
    LifecyclePanelResult,
    SecurityLifecycle,
    TerminalAction,
    apply_lifecycle,
    apply_lifecycle_panel,
    lifecycle_panel_report,
    lifecycle_returns,
    tradeable_mask,
)


# ---------- SecurityLifecycle -----------------------------------------


def test_dataclass_defaults():
    lc = SecurityLifecycle(symbol="X")
    assert lc.symbol == "X"
    assert lc.listing_date is None
    assert lc.delisting_date is None
    assert lc.terminal_action == TerminalAction.MARK_TO_ZERO
    assert lc.terminal_recovery == 0.0


def test_rejects_listing_after_delisting():
    with pytest.raises(KuantValueError):
        SecurityLifecycle(
            symbol="X",
            listing_date=date(2020, 6, 1),
            delisting_date=date(2020, 1, 1),
        )


def test_rejects_recovery_out_of_range():
    with pytest.raises(KuantValueError):
        SecurityLifecycle(symbol="X", terminal_recovery=1.5)


def test_summary_contains_symbol():
    lc = SecurityLifecycle(symbol="ACME")
    assert "ACME" in lc.summary()


# ---------- tradeable_mask --------------------------------------------


def test_mask_open_ends_all_true():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lc = SecurityLifecycle(symbol="X")
    m = tradeable_mask(idx, lc)
    assert m.all()
    assert m.dtype == bool
    assert m.shape == (5,)


def test_mask_masks_pre_listing():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lc = SecurityLifecycle(symbol="X", listing_date=date(2020, 1, 3))
    m = tradeable_mask(idx, lc)
    assert m.tolist() == [False, False, True, True, True]


def test_mask_masks_post_delisting():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lc = SecurityLifecycle(symbol="X", delisting_date=date(2020, 1, 3))
    m = tradeable_mask(idx, lc)
    assert m.tolist() == [True, True, True, False, False]


def test_mask_both_ends():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    lc = SecurityLifecycle(
        symbol="X",
        listing_date=date(2020, 1, 2),
        delisting_date=date(2020, 1, 4),
    )
    m = tradeable_mask(idx, lc)
    assert m.tolist() == [False, True, True, True, False]


# ---------- apply_lifecycle -------------------------------------------


def test_apply_masks_series():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    p = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
    lc = SecurityLifecycle(
        symbol="X",
        listing_date=date(2020, 1, 2),
        delisting_date=date(2020, 1, 4),
    )
    out = apply_lifecycle(p, lc)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 11.0
    assert out.iloc[3] == 13.0
    assert np.isnan(out.iloc[4])


def test_apply_rejects_non_series():
    lc = SecurityLifecycle(symbol="X")
    with pytest.raises(KuantShapeError):
        apply_lifecycle(np.arange(5.0), lc)


def test_apply_panel_masks_columns():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0],
            "B": [20.0, 21.0, 22.0, 23.0, 24.0],
        },
        index=idx,
    )
    lcs = {
        "A": SecurityLifecycle(symbol="A", delisting_date=date(2020, 1, 3)),
    }
    out = apply_lifecycle_panel(df, lcs)
    assert out["A"].iloc[2] == 12.0
    assert np.isnan(out["A"].iloc[3])
    # B untouched.
    assert out["B"].iloc[4] == 24.0


def test_apply_panel_ignores_missing_columns():
    """Lifecycle for a symbol not in the panel is silently ignored."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    lcs = {
        "A": SecurityLifecycle(symbol="A"),
        "GONE": SecurityLifecycle(symbol="GONE", delisting_date=date(2019, 1, 1)),
    }
    out = apply_lifecycle_panel(df, lcs)
    assert "GONE" not in out.columns


def test_apply_panel_rejects_non_dataframe():
    with pytest.raises(KuantShapeError):
        apply_lifecycle_panel(np.zeros((3, 2)), {})


# ---------- lifecycle_returns -----------------------------------------


def test_returns_mark_to_zero():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    p = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
    lc = SecurityLifecycle(
        symbol="X",
        delisting_date=date(2020, 1, 3),
        terminal_action=TerminalAction.MARK_TO_ZERO,
    )
    ret = lifecycle_returns(p, lc)
    # In-window rows: pct_change of 10, 11, 12
    assert np.isnan(ret.iloc[0])
    assert ret.iloc[1] == pytest.approx(0.1)
    assert ret.iloc[2] == pytest.approx(12 / 11 - 1)
    # Terminal-day-plus-one: -1.0
    assert ret.iloc[3] == pytest.approx(-1.0)
    # Later: NaN.
    assert np.isnan(ret.iloc[4])


def test_returns_liquidate_at_last():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    p = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
    lc = SecurityLifecycle(
        symbol="X",
        delisting_date=date(2020, 1, 3),
        terminal_action=TerminalAction.LIQUIDATE_AT_LAST,
    )
    ret = lifecycle_returns(p, lc)
    assert ret.iloc[3] == pytest.approx(0.0)
    assert np.isnan(ret.iloc[4])


def test_returns_prorate_recovery():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    p = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
    lc = SecurityLifecycle(
        symbol="X",
        delisting_date=date(2020, 1, 3),
        terminal_action=TerminalAction.PRORATE_RECOVERY,
        terminal_recovery=0.30,
    )
    ret = lifecycle_returns(p, lc)
    assert ret.iloc[3] == pytest.approx(0.30 - 1.0)


def test_returns_no_delisting_date():
    """No delisting date → returns are ordinary pct_change."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    p = pd.Series([10.0, 11.0, 12.0, 13.0], index=idx)
    lc = SecurityLifecycle(symbol="X")
    ret = lifecycle_returns(p, lc)
    assert ret.iloc[1] == pytest.approx(0.1)
    assert np.isfinite(ret.iloc[3])


def test_returns_delisting_at_end_of_series_no_terminal_row():
    """When delisting_date is on the last index date, no post-delist
    row exists to carry the terminal transition."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    p = pd.Series([10.0, 11.0, 12.0], index=idx)
    lc = SecurityLifecycle(
        symbol="X",
        delisting_date=date(2020, 1, 3),
        terminal_action=TerminalAction.MARK_TO_ZERO,
    )
    ret = lifecycle_returns(p, lc)
    # No row after 2020-01-03 exists to carry the terminal return.
    assert ret.iloc[2] == pytest.approx(12 / 11 - 1)


# ---------- LifecyclePanelResult -------------------------------------


def test_panel_report_shape():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"A": [10.0, 11.0, 12.0, 13.0, 14.0], "B": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=idx,
    )
    lcs = {
        "A": SecurityLifecycle(symbol="A", delisting_date=date(2020, 1, 3)),
    }
    r = lifecycle_panel_report(df, lcs)
    assert isinstance(r, LifecyclePanelResult)
    assert r.cleaned.shape == (5, 2)
    assert r.tradeable.shape == (5, 1)
    assert r.terminal_returns.iloc[3]["A"] == -1.0


def test_panel_result_summary_contains_shape():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    r = lifecycle_panel_report(df, {"A": SecurityLifecycle(symbol="A")})
    s = r.summary()
    assert "3" in s


def test_panel_result_to_parquet_roundtrip(tmp_path):
    """`.to_parquet` writes a readable file with a row_index column."""
    pq = pytest.importorskip("pyarrow.parquet")
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    r = lifecycle_panel_report(df, {"A": SecurityLifecycle(symbol="A")})
    path = tmp_path / "lc.parquet"
    r.to_parquet(path)
    table = pq.read_table(path)
    cols = table.column_names
    assert "row_index" in cols
    assert "A" in cols

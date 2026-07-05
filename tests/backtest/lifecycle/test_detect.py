"""Tests for kuant.backtest.lifecycle.detect."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.backtest.lifecycle.detect import detect_delistings, lifecycles_from_panel
from kuant.backtest.lifecycle.security import SecurityLifecycle, TerminalAction
from kuant.errors import KuantShapeError, KuantValueError


def test_detects_column_with_trailing_nans():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "LIVE": np.arange(20, dtype=float),
            "GONE": list(range(10)) + [np.nan] * 10,
        },
        index=idx,
    )
    out = detect_delistings(df, min_gap_days=5)
    assert "GONE" in out
    assert "LIVE" not in out
    assert out["GONE"] == date(2020, 1, 10)


def test_short_nan_gap_not_flagged():
    """A 3-NaN trailing tail with min_gap_days=5 should NOT flag."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"HALTED": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan, np.nan]},
        index=idx,
    )
    out = detect_delistings(df, min_gap_days=5)
    assert out == {}


def test_all_nan_column_not_flagged():
    """A column that never printed should not be flagged."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"NEVER": [np.nan] * 10}, index=idx)
    out = detect_delistings(df, min_gap_days=5)
    assert out == {}


def test_still_live_at_series_end_not_flagged():
    """A column with a last valid at the last index date is not delisted."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"LIVE": np.arange(10, dtype=float)}, index=idx)
    out = detect_delistings(df, min_gap_days=5)
    assert out == {}


def test_rejects_non_dataframe():
    with pytest.raises(KuantShapeError):
        detect_delistings(np.zeros((5, 2)))


def test_rejects_bad_min_gap_days():
    df = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(KuantValueError):
        detect_delistings(df, min_gap_days=0)


def test_lifecycles_from_panel_wraps_detection():
    idx = pd.date_range("2020-01-01", periods=15, freq="D")
    df = pd.DataFrame(
        {
            "LIVE": np.arange(15, dtype=float),
            "GONE": list(range(5)) + [np.nan] * 10,
        },
        index=idx,
    )
    lcs = lifecycles_from_panel(df, min_gap_days=5)
    assert set(lcs.keys()) == {"GONE"}
    lc = lcs["GONE"]
    assert isinstance(lc, SecurityLifecycle)
    assert lc.delisting_date == date(2020, 1, 5)
    assert lc.terminal_action == TerminalAction.MARK_TO_ZERO


def test_lifecycles_from_panel_respects_terminal_action():
    idx = pd.date_range("2020-01-01", periods=15, freq="D")
    df = pd.DataFrame(
        {"GONE": list(range(5)) + [np.nan] * 10},
        index=idx,
    )
    lcs = lifecycles_from_panel(
        df, min_gap_days=5, terminal_action=TerminalAction.LIQUIDATE_AT_LAST
    )
    assert lcs["GONE"].terminal_action == TerminalAction.LIQUIDATE_AT_LAST


def test_detect_multiple_columns_returns_all():
    idx = pd.date_range("2020-01-01", periods=15, freq="D")
    df = pd.DataFrame(
        {
            "A": list(range(3)) + [np.nan] * 12,
            "B": list(range(7)) + [np.nan] * 8,
            "C": np.arange(15, dtype=float),
        },
        index=idx,
    )
    out = detect_delistings(df, min_gap_days=5)
    assert set(out.keys()) == {"A", "B"}
    assert out["A"] == date(2020, 1, 3)
    assert out["B"] == date(2020, 1, 7)

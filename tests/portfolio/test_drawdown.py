"""Tests for kuant.portfolio.drawdown."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.portfolio.drawdown import DrawdownResult, drawdown


# ---------- basic mechanics ---------------------------------------------


def test_max_dd_on_known_curve():
    """Peak at 110 (index 2), trough at 88 (index 4). Max DD = -20%."""
    eq = np.array([100.0, 105, 110, 100, 88, 95, 105])
    r = drawdown(eq)
    assert abs(r.max_dd - (-0.20)) < 1e-9
    assert r.peak_position == 2
    assert r.trough_position == 4
    assert r.duration == 2


def test_series_zero_at_new_peaks():
    """At every new equity high the drawdown is 0."""
    eq = np.array([100.0, 105, 110, 115, 120])
    r = drawdown(eq)
    assert r.series.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert r.max_dd == 0.0


def test_monotone_decline_gives_running_max_at_first_bar():
    """A curve that only falls: peak stays at index 0."""
    eq = np.array([100.0, 95, 90, 85])
    r = drawdown(eq)
    assert r.peak_position == 0
    assert r.trough_position == 3
    assert abs(r.max_dd - (-0.15)) < 1e-9


# ---------- recovery flag ------------------------------------------------


def test_recovered_true_when_new_high_after_trough():
    eq = np.array([100.0, 110, 90, 105, 120])
    r = drawdown(eq)
    # Trough at index 2, then reaches 105 > peak 110? No — reaches 120 which
    # is > peak 110, so recovered.
    assert r.recovered


def test_recovered_false_when_still_underwater_at_end():
    eq = np.array([100.0, 110, 90, 95, 100])  # never reaches 110 again
    r = drawdown(eq)
    assert r.recovered is False


# ---------- NaN handling -------------------------------------------------


def test_nan_bars_produce_nan_drawdown():
    eq = np.array([100.0, np.nan, 110, 100])
    r = drawdown(eq)
    assert np.isnan(r.series[1])
    # Finite bars should still produce valid drawdowns.
    assert r.series[0] == 0.0
    assert r.series[2] == 0.0


def test_leading_nan_positions_have_nan_drawdown():
    """Before the first finite bar there is no running max."""
    eq = np.array([np.nan, np.nan, 100, 110, 100])
    r = drawdown(eq)
    assert np.isnan(r.series[:2]).all()
    assert r.series[2] == 0.0
    assert r.series[3] == 0.0
    # Trough is at index 4 (equity 100 vs peak 110 → -1/11).
    assert abs(r.series[4] - (-1 / 11)) < 1e-9


# ---------- error contract -----------------------------------------------


def test_reject_non_positive_equity():
    eq = np.array([100.0, 105, 0.0, 90])
    with pytest.raises(KuantValueError):
        drawdown(eq)


def test_reject_negative_equity():
    eq = np.array([100.0, 105, -50.0, 90])
    with pytest.raises(KuantValueError):
        drawdown(eq)


def test_reject_2d_input():
    with pytest.raises(Exception):
        drawdown(np.zeros((5, 3)))


# ---------- empty + all-NaN edge cases -----------------------------------


def test_empty_input_raises():
    from kuant.errors import KuantValueError

    with pytest.raises(KuantValueError, match="KE-VAL-EMPTY"):
        drawdown(np.array([]))


def test_all_nan_input_yields_nan_max_dd():
    r = drawdown(np.array([np.nan, np.nan]))
    assert np.isnan(r.max_dd)


# ---------- result contract ----------------------------------------------


def test_returns_dataclass():
    r = drawdown(np.arange(1.0, 5))
    assert isinstance(r, DrawdownResult)


def test_summary_contains_metadata():
    r = drawdown(np.array([100.0, 90, 105]))
    s = r.summary()
    assert "DrawdownResult" in s
    assert "max drawdown" in s


def test_to_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    r = drawdown(np.array([100.0, 90, 105, 95]))
    path = tmp_path / "dd.parquet"
    r.to_parquet(path)
    cols = pq.read_table(path).column_names
    assert cols == ["drawdown"]


# ---------- consistency with running-max formula -------------------------


def test_formula_matches_manual_computation():
    rng = np.random.default_rng(0)
    eq = 100 + np.cumsum(rng.normal(0.05, 1.0, 500))
    eq = np.maximum(eq, 1.0)  # keep positive
    r = drawdown(eq)
    manual = eq / np.maximum.accumulate(eq) - 1
    assert np.allclose(r.series, manual)
    assert abs(r.max_dd - float(manual.min())) < 1e-12

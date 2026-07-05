"""Tests for kuant.edgecases.delistedhandling."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.edgecases.delistedhandling import (
    RecoveryCheckResult,
    full_recovery_check,
    hold_last_price,
    zero_after_delist,
)
from kuant.errors import KuantNumericWarning, KuantValueError


# ---------- zero_after_delist -------------------------------------------


def test_zero_after_delist_basic():
    prices = np.array([100.0, 95, 90, 80, 70])
    out = zero_after_delist(prices, 3)
    assert out.tolist() == [100.0, 95.0, 90.0, 0.0, 0.0]


def test_zero_after_delist_at_zero_makes_all_zero():
    prices = np.array([100.0, 95, 90])
    out = zero_after_delist(prices, 0)
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_zero_after_delist_at_end_makes_no_change():
    prices = np.array([100.0, 95, 90])
    out = zero_after_delist(prices, 3)  # position == n → nothing zeroed
    assert out.tolist() == prices.tolist()


def test_zero_after_delist_does_not_mutate_input():
    prices = np.array([100.0, 95, 90])
    _ = zero_after_delist(prices, 1)
    assert prices.tolist() == [100.0, 95.0, 90.0]


# ---------- hold_last_price ---------------------------------------------


def test_hold_last_price_basic():
    prices = np.array([100.0, 95, 90, 80, 70])
    out = hold_last_price(prices, 3, max_hold_days=100)
    assert out.tolist() == [100.0, 95.0, 90.0, 90.0, 90.0]


def test_hold_last_price_at_position_zero_holds_first_row():
    prices = np.array([100.0, 95, 90])
    out = hold_last_price(prices, 0, max_hold_days=100)
    assert out.tolist() == [100.0, 100.0, 100.0]


def test_hold_last_price_warns_on_long_hold():
    """Phantom-equity signal: held span exceeds max_hold_days."""
    prices = np.arange(50.0)
    with pytest.warns(KuantNumericWarning) as record:
        hold_last_price(prices, delist_position=10, max_hold_days=20)
    assert any("KW-PHANTOM-EQUITY" in str(w.message) for w in record)


def test_hold_last_price_no_warning_under_threshold():
    """Held span within max_hold_days should NOT warn."""
    prices = np.arange(30.0)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", KuantNumericWarning)
        # 30 - 20 = 10 day hold, well under default max_hold_days=20.
        hold_last_price(prices, delist_position=20, max_hold_days=20)


def test_hold_last_price_at_end_holds_nothing():
    """delist_position == n → no rows to fill."""
    prices = np.array([100.0, 95, 90])
    out = hold_last_price(prices, 3, max_hold_days=100)
    assert out.tolist() == prices.tolist()


# ---------- error contract on positions ----------------------------------


def test_zero_after_delist_rejects_out_of_bounds():
    with pytest.raises(KuantValueError):
        zero_after_delist(np.arange(5.0), 10)


def test_hold_last_price_rejects_negative_position():
    with pytest.raises(KuantValueError):
        hold_last_price(np.arange(5.0), -1)


def test_hold_last_price_rejects_zero_max_hold_days():
    with pytest.raises(KuantValueError):
        hold_last_price(np.arange(5.0), 2, max_hold_days=0)


def test_zero_after_delist_rejects_2d_prices():
    with pytest.raises(Exception):
        zero_after_delist(np.zeros((5, 3)), 2)


def test_zero_after_delist_rejects_non_integer_position():
    with pytest.raises(KuantValueError):
        zero_after_delist(np.arange(5.0), 2.5)


# ---------- full_recovery_check -----------------------------------------


def test_recovery_check_full_coverage_is_clean():
    universe = np.array(["AAPL", "MSFT", "ENRN", "LEH", "WCOM"])
    known = np.array(["ENRN", "LEH", "WCOM"])
    r = full_recovery_check(universe, known, tolerance=0.9)
    assert r.status == "clean"
    assert r.coverage == 1.0
    assert r.missing == []


def test_recovery_check_missing_names_flag_survivor_bias():
    """Some known-delisted names missing → survivor_bias status + warning."""
    universe = np.array(["AAPL", "MSFT", "ENRN"])
    known = np.array(["ENRN", "LEH", "WCOM"])
    with pytest.warns(KuantNumericWarning) as record:
        r = full_recovery_check(universe, known, tolerance=0.9)
    assert r.status == "survivor_bias"
    assert r.coverage < 0.9
    assert set(r.missing) == {"LEH", "WCOM"}
    assert any("KW-SURVIVOR-BIAS" in str(w.message) for w in record)


def test_recovery_check_returns_dataclass_type():
    universe = np.array(["A", "B"])
    known = np.array(["A"])
    r = full_recovery_check(universe, known)
    assert isinstance(r, RecoveryCheckResult)


def test_recovery_check_summary_contains_metadata():
    universe = np.array(["A", "B"])
    known = np.array(["A", "C"])
    r = full_recovery_check(universe, known, tolerance=0.9)
    s = r.summary()
    assert "RecoveryCheckResult" in s
    assert "coverage" in s
    assert "tolerance" in s


def test_recovery_check_rejects_empty_known():
    with pytest.raises(KuantValueError):
        full_recovery_check(np.array(["A"]), np.array([]))


def test_recovery_check_missing_sorted():
    """Missing names should come back sorted for reproducibility."""
    universe = np.array(["A"])
    known = np.array(["Z", "M", "B"])
    r = full_recovery_check(universe, known)
    assert r.missing == ["B", "M", "Z"]


def test_recovery_check_low_tolerance_accepts_partial():
    """A low tolerance lets partial coverage still be 'clean'."""
    universe = np.array(["A"])
    known = np.array(["A", "B"])
    r = full_recovery_check(universe, known, tolerance=0.4)
    assert r.status == "clean"
    assert r.coverage == 0.5

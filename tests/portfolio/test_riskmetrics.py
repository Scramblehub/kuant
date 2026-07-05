"""Tests for kuant.portfolio.riskmetrics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from kuant.errors import KuantValueError  # noqa: E402
from kuant.portfolio.riskmetrics import (  # noqa: E402
    DrawdownTableResult,
    UlcerResult,
    deflated_sharpe,
    down_capture,
    drawdown_table,
    kelly,
    omega,
    probabilistic_sharpe,
    ulcer_index,
    up_capture,
)


# ---------- omega ------------------------------------------------------


def test_omega_greater_than_1_for_positive_drift():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 2000)
    assert omega(r) > 1.0


def test_omega_less_than_1_for_negative_drift():
    rng = np.random.default_rng(0)
    r = rng.normal(-0.001, 0.01, 2000)
    assert omega(r) < 1.0


def test_omega_all_positive_returns_infinity():
    r = np.array([0.01, 0.02, 0.03])
    assert omega(r) == float("inf")


def test_omega_all_negative_below_threshold_zero():
    r = np.array([-0.01, -0.02, -0.03])
    assert omega(r) == 0.0


def test_omega_threshold_shifts_boundary():
    """A higher threshold makes more returns 'downside' and lowers Omega."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    low = omega(r, threshold=-0.005)
    high = omega(r, threshold=0.005)
    assert low > high


def test_omega_rejects_all_nan():
    with pytest.raises(KuantValueError):
        omega(np.array([np.nan, np.nan]))


# ---------- ulcer index -----------------------------------------------


def test_ulcer_index_zero_on_monotone_uptrend():
    eq = np.array([100.0, 101, 102, 103])
    r = ulcer_index(eq)
    assert r.ulcer_index == 0.0


def test_ulcer_index_positive_on_drawdown():
    eq = np.array([100.0, 90, 95, 100, 90, 100])
    r = ulcer_index(eq)
    assert r.ulcer_index > 0.0


def test_ulcer_index_rejects_non_positive_equity():
    with pytest.raises(KuantValueError):
        ulcer_index(np.array([100.0, 0.0, 50]))


def test_ulcer_result_type():
    assert isinstance(ulcer_index(np.array([100.0, 90, 100])), UlcerResult)


# ---------- kelly criterion -------------------------------------------


def test_kelly_positive_expected_return_positive_fraction():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    assert kelly(r) > 0


def test_kelly_negative_expected_return_zero():
    rng = np.random.default_rng(0)
    r = rng.normal(-0.001, 0.01, 1000)
    assert kelly(r) == 0.0


def test_kelly_cap_enforced():
    """Kelly can be very large for tight-var samples; cap holds."""
    # Small variance → large raw Kelly.
    r = np.array([0.01, 0.011, 0.012, 0.010, 0.011])
    assert kelly(r, cap=0.5) <= 0.5


def test_kelly_rejects_cap_out_of_range():
    with pytest.raises(KuantValueError):
        kelly(np.arange(10.0), cap=1.5)


def test_kelly_constant_returns_zero():
    """Zero variance → Kelly = 0 by convention, with a warning."""
    from kuant.errors import KuantNumericWarning

    r = np.full(100, 0.01)
    with pytest.warns(KuantNumericWarning, match="KW-KELLY-ZERO-VARIANCE"):
        assert kelly(r) == 0.0


# ---------- up / down capture -----------------------------------------


def test_up_capture_of_identical_strategy_is_one():
    """A strategy that IS the benchmark has capture = 1."""
    rng = np.random.default_rng(0)
    bench = rng.normal(0.001, 0.01, 500)
    assert abs(up_capture(bench, bench) - 1.0) < 1e-9
    assert abs(down_capture(bench, bench) - 1.0) < 1e-9


def test_up_capture_leveraged_greater_than_one():
    rng = np.random.default_rng(0)
    bench = rng.normal(0.001, 0.01, 500)
    strategy = 2.0 * bench
    assert abs(up_capture(strategy, bench) - 2.0) < 1e-9


def test_capture_rejects_length_mismatch():
    with pytest.raises(Exception):
        up_capture(np.arange(5.0), np.arange(6.0))


# ---------- probabilistic + deflated Sharpe --------------------------


def test_psr_high_sharpe_high_probability():
    """A Sharpe of 2.0 on 500 obs should give a very high PSR."""
    assert probabilistic_sharpe(sharpe=2.0, n=500) > 0.99


def test_psr_zero_sharpe_50_50():
    """A Sharpe of 0 vs benchmark 0 should give ~0.5."""
    p = probabilistic_sharpe(sharpe=0.0, n=252)
    assert abs(p - 0.5) < 0.01


def test_psr_negative_sharpe_low_probability():
    assert probabilistic_sharpe(sharpe=-1.0, n=252) < 0.05


def test_psr_rejects_zero_n():
    with pytest.raises(KuantValueError):
        probabilistic_sharpe(sharpe=1.0, n=0)


def test_deflated_sharpe_penalizes_more_trials():
    """More trials tested → lower DSR for the same observed Sharpe."""
    dsr_1 = deflated_sharpe(sharpe=2.0, n=500, n_trials=2, variance_of_sharpes=1.0)
    dsr_1000 = deflated_sharpe(sharpe=2.0, n=500, n_trials=1000, variance_of_sharpes=1.0)
    assert dsr_1 > dsr_1000


def test_deflated_sharpe_single_trial_equals_psr():
    """With n_trials=1 the deflator collapses to PSR@0."""
    psr = probabilistic_sharpe(sharpe=1.5, n=500, sharpe_benchmark=0.0)
    dsr = deflated_sharpe(sharpe=1.5, n=500, n_trials=1, variance_of_sharpes=0.0)
    assert abs(dsr - psr) < 1e-6


# ---------- drawdown table -------------------------------------------


def test_drawdown_table_finds_main_episode():
    """Simple: single episode peak→trough→recovery."""
    eq = np.array([100.0, 110, 120, 100, 80, 90, 100, 130])
    r = drawdown_table(eq, top_n=1)
    assert r.n == 1
    # Peak 120 at index 2, trough 80 at index 4, recovery at index 7.
    assert r.peaks[0] == 2
    assert r.troughs[0] == 4
    assert r.recoveries[0] == 7
    assert abs(r.depths[0] - (80 / 120 - 1)) < 1e-9


def test_drawdown_table_still_underwater_no_recovery():
    eq = np.array([100.0, 110, 100, 90, 85])
    r = drawdown_table(eq, top_n=1)
    assert r.recoveries[0] is None
    assert r.recovery_times[0] is None


def test_drawdown_table_multiple_episodes_sorted_by_depth():
    """Two separate episodes: -10% then -30%. Table sorted by depth."""
    eq = np.array([100.0, 110, 105, 99, 110, 120, 100, 84, 120])
    r = drawdown_table(eq, top_n=2)
    assert r.n == 2
    # Deepest first.
    assert r.depths[0] < r.depths[1]


def test_drawdown_table_top_n_caps_output():
    """Even if more episodes exist, only top_n rows are returned."""
    rng = np.random.default_rng(0)
    r_returns = rng.normal(0.0005, 0.02, 500)
    eq = 100 * np.cumprod(1 + r_returns)
    res = drawdown_table(eq, top_n=3)
    assert res.n <= 3


def test_drawdown_table_returns_dataclass():
    eq = np.array([100.0, 90, 95, 100])
    assert isinstance(drawdown_table(eq), DrawdownTableResult)


def test_drawdown_table_rejects_zero_top_n():
    with pytest.raises(KuantValueError):
        drawdown_table(np.array([100.0, 90]), top_n=0)


def test_drawdown_table_summary_string():
    eq = np.array([100.0, 110, 120, 100, 80, 90, 100, 130])
    s = drawdown_table(eq).summary()
    assert "DrawdownTableResult" in s


# ---------- summary render (smoke) -----------------------------------


def test_ulcer_summary_string():
    r = ulcer_index(np.array([100.0, 90, 95, 100]))
    assert "UlcerResult" in r.summary()

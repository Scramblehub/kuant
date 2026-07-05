"""Tests for kuant.backtest.warmup."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.backtest.lifecycle import SecurityLifecycle, TerminalAction
from kuant.backtest.liquidity import LiquidityProfile
from kuant.backtest.warmup import Warmup, WarmupMode
from kuant.errors import KuantShapeError, KuantValueError


def _panel(n: int = 100, symbols=("A", "B")) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {}
    for i, sym in enumerate(symbols):
        data[sym] = np.linspace(50 + i * 10, 100 + i * 10, n)
    return pd.DataFrame(data, index=idx)


def _rolling_mean_np(x, window):
    """Simple rolling mean kernel returning a 1D np.ndarray."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if window > len(x):
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


# ---------- construction ------------------------------------------------


def test_construct_defaults_to_eager():
    w = Warmup(_panel())
    cache = w.materialize()
    assert cache.mode is WarmupMode.EAGER


def test_construct_accepts_string_mode():
    w = Warmup(_panel(), mode="lazy")
    cache = w.materialize()
    assert cache.mode is WarmupMode.LAZY


def test_construct_rejects_bad_mode_string():
    with pytest.raises(KuantValueError):
        Warmup(_panel(), mode="bogus")


def test_construct_rejects_non_dataframe():
    with pytest.raises(KuantShapeError):
        Warmup(np.zeros((5, 3)))


# ---------- add_indicator + EAGER materialization ---------------------


def test_add_indicator_per_symbol_materializes_eagerly():
    w = Warmup(_panel(), mode="eager")
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    assert cache.is_cached("sma5")
    val = cache.get("sma5", timestamp=date(2020, 1, 10), symbol="A")
    assert np.isfinite(val)


def test_get_returns_series_when_symbol_omitted():
    w = Warmup(_panel(), mode="eager")
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    row = cache.get("sma5", timestamp=date(2020, 1, 10))
    assert isinstance(row, pd.Series)
    assert "A" in row.index and "B" in row.index


def test_get_rejects_unknown_indicator():
    w = Warmup(_panel(), mode="eager")
    cache = w.materialize()
    with pytest.raises(KuantValueError):
        cache.get("nonexistent", timestamp=date(2020, 1, 10))


def test_get_rejects_unknown_symbol():
    w = Warmup(_panel(), mode="eager")
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    with pytest.raises(KuantValueError):
        cache.get("sma5", timestamp=date(2020, 1, 10), symbol="ZZZ")


def test_duplicate_indicator_name_rejected():
    w = Warmup(_panel())
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    with pytest.raises(KuantValueError):
        w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=10)


# ---------- LAZY mode -------------------------------------------------


def test_lazy_mode_not_cached_until_first_access():
    w = Warmup(_panel(), mode="lazy")
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    assert not cache.is_cached("sma5")
    _ = cache.get("sma5", timestamp=date(2020, 1, 10), symbol="A")
    assert cache.is_cached("sma5")


def test_lazy_mode_second_access_uses_cache():
    """This is a behavioural not-a-timing test: the result is stable
    across two identical .get() calls."""
    w = Warmup(_panel(), mode="lazy")
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    a = cache.get("sma5", timestamp=date(2020, 1, 10), symbol="A")
    b = cache.get("sma5", timestamp=date(2020, 1, 10), symbol="A")
    assert a == b


# ---------- OFF mode --------------------------------------------------


def test_off_mode_never_caches():
    w = Warmup(_panel(), mode="off")
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    assert not cache.is_cached("sma5")
    _ = cache.get("sma5", timestamp=date(2020, 1, 10), symbol="A")
    assert not cache.is_cached("sma5")


def test_off_mode_still_returns_correct_values():
    """Turning off caching should not change correctness."""
    w_off = Warmup(_panel(), mode="off")
    w_off.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache_off = w_off.materialize()

    w_eager = Warmup(_panel(), mode="eager")
    w_eager.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache_eager = w_eager.materialize()

    val_off = cache_off.get("sma5", timestamp=date(2020, 1, 20), symbol="A")
    val_eager = cache_eager.get("sma5", timestamp=date(2020, 1, 20), symbol="A")
    assert val_off == val_eager


# ---------- per-indicator cache override ------------------------------


def test_cache_override_forces_cache_in_off_mode():
    w = Warmup(_panel(), mode="off")
    w.add_indicator("live", _rolling_mean_np, per_symbol=True, window=5)
    w.add_indicator("cached", _rolling_mean_np, per_symbol=True, cache=True, window=10)
    cache = w.materialize()
    # In OFF mode, `cached` is still materialized because of the override.
    assert cache.is_cached("cached")
    assert not cache.is_cached("live")


def test_cache_override_disables_cache_in_eager_mode():
    w = Warmup(_panel(), mode="eager")
    w.add_indicator("cached", _rolling_mean_np, per_symbol=True, window=5)
    w.add_indicator("live", _rolling_mean_np, per_symbol=True, cache=False, window=10)
    cache = w.materialize()
    assert cache.is_cached("cached")
    assert not cache.is_cached("live")


# ---------- universe membership ---------------------------------------


def test_universe_membership_gate():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5}, index=idx)
    membership = pd.DataFrame(
        {"A": [True] * 5, "B": [True, True, False, False, True]},
        index=idx,
    )
    w = Warmup(prices)
    w.add_universe_membership(membership)
    cache = w.materialize()
    assert cache.universe(date(2020, 1, 1), "A") is True
    assert cache.universe(date(2020, 1, 3), "B") is False
    assert cache.universe(date(2020, 1, 5), "B") is True


def test_universe_defaults_true_when_not_registered():
    w = Warmup(_panel())
    cache = w.materialize()
    assert cache.universe(date(2020, 1, 1), "A") is True


def test_membership_length_mismatch_rejected():
    prices = _panel(n=10)
    membership = pd.DataFrame({"A": [True] * 5}, index=pd.date_range("2020-01-01", periods=5))
    w = Warmup(prices)
    with pytest.raises(KuantShapeError):
        w.add_universe_membership(membership)


# ---------- lifecycle panel ------------------------------------------


def test_lifecycle_panel_masks_post_delisting():
    prices = _panel(n=10)
    lc = SecurityLifecycle(
        symbol="A",
        delisting_date=date(2020, 1, 5),
        terminal_action=TerminalAction.MARK_TO_ZERO,
    )
    w = Warmup(prices)
    w.add_lifecycles({"A": lc})
    cache = w.materialize()
    assert cache.tradeable(date(2020, 1, 5), "A") is True
    assert cache.tradeable(date(2020, 1, 6), "A") is False
    # B has no lifecycle → always tradeable.
    assert cache.tradeable(date(2020, 1, 9), "B") is True


def test_lifecycle_rejects_non_lifecycle_value():
    w = Warmup(_panel())
    with pytest.raises(KuantValueError):
        w.add_lifecycles({"A": "not a lifecycle"})


# ---------- liquidity panel ------------------------------------------


def test_liquidity_panel_masks_zero_adv():
    prices = _panel(n=5)
    idx = prices.index
    adv = pd.Series([1e6, 1e6, 0.0, 1e6, 1e6], index=idx)
    profile = LiquidityProfile(symbol="A", adv_series=adv, min_size=1.0)
    w = Warmup(prices)
    w.add_liquidity_profiles({"A": profile})
    cache = w.materialize()
    assert cache.liquid(date(2020, 1, 1), "A") is True
    assert cache.liquid(date(2020, 1, 3), "A") is False
    # B has no profile → always liquid.
    assert cache.liquid(date(2020, 1, 3), "B") is True


def test_liquidity_rejects_non_profile_value():
    w = Warmup(_panel())
    with pytest.raises(KuantValueError):
        w.add_liquidity_profiles({"A": "not a profile"})


# ---------- summary + timing -----------------------------------------


def test_materialization_time_recorded():
    w = Warmup(_panel(), mode="eager")
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    assert cache.materialization_time_s >= 0.0


def test_summary_contains_mode_and_shape():
    w = Warmup(_panel(n=50))
    w.add_indicator("sma5", _rolling_mean_np, per_symbol=True, window=5)
    cache = w.materialize()
    s = cache.summary()
    assert "eager" in s
    assert "(50, 2)" in s


# ---------- panel-mode indicator (kernel takes DataFrame) ------------


def test_panel_mode_indicator():
    """Kernel that consumes the whole DataFrame at once."""

    def _panel_mean(df):
        return df.rolling(window=5, min_periods=5).mean()

    w = Warmup(_panel(), mode="eager")
    w.add_indicator("panelmean5", _panel_mean, per_symbol=False)
    cache = w.materialize()
    val = cache.get("panelmean5", timestamp=date(2020, 1, 10), symbol="A")
    assert np.isfinite(val)


# ---------- integration: lifecycle + liquidity + universe -------------


def test_all_gates_compose():
    prices = _panel(n=10)
    idx = prices.index
    membership = pd.DataFrame({"A": [True] * 10, "B": [True] * 5 + [False] * 5}, index=idx)
    lc_a = SecurityLifecycle(
        symbol="A",
        delisting_date=date(2020, 1, 7),
        terminal_action=TerminalAction.MARK_TO_ZERO,
    )
    prof_a = LiquidityProfile(
        symbol="A",
        adv_series=pd.Series([1e6, 1e6, 0.0, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6], index=idx),
        min_size=1.0,
    )
    w = Warmup(prices)
    w.add_universe_membership(membership)
    w.add_lifecycles({"A": lc_a})
    w.add_liquidity_profiles({"A": prof_a})
    cache = w.materialize()

    # 2020-01-01 A: in-universe + tradeable + liquid.
    assert cache.universe(date(2020, 1, 1), "A")
    assert cache.tradeable(date(2020, 1, 1), "A")
    assert cache.liquid(date(2020, 1, 1), "A")
    # 2020-01-03 A: liquidity is out.
    assert not cache.liquid(date(2020, 1, 3), "A")
    # 2020-01-08 A: post-delisting.
    assert not cache.tradeable(date(2020, 1, 8), "A")
    # 2020-01-06 B: dropped from universe.
    assert not cache.universe(date(2020, 1, 6), "B")

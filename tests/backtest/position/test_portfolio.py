"""Tests for kuant.backtest.position.PortfolioState + EquitySnapshot."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.backtest.fill import Order, OrderSide, submit_order
from kuant.backtest.liquidity import FlatSlippage, LiquidityProfile
from kuant.backtest.position import EquitySnapshot, PortfolioState
from kuant.errors import KuantValueError


def _profile(symbol: str = "XYZ", adv: float = 1_000_000.0) -> LiquidityProfile:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    adv_s = pd.Series([adv] * 5, index=idx, dtype=float)
    return LiquidityProfile(symbol=symbol, adv_series=adv_s, min_size=1.0, max_participation=0.10)


def _make_report(symbol, side, size, ts, price, model):
    order = Order(symbol=symbol, side=side, size=size, timestamp=ts)
    profile = _profile(symbol=symbol)
    return submit_order(order, profile, price=price, model=model)


# ---------- basic apply_fill flow -------------------------------------


def test_starting_state_zero_cash_by_default():
    ps = PortfolioState()
    assert ps.cash == 0.0
    assert ps.positions == {}


def test_apply_buy_debits_cash_and_creates_position():
    ps = PortfolioState(cash=100_000.0)
    report = _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    ps.apply_fill(report)
    assert ps.cash == pytest.approx(100_000.0 - 100 * 50.0)
    assert ps.positions["XYZ"].size == 100.0


def test_apply_sell_credits_cash():
    ps = PortfolioState(cash=100_000.0)
    ps.apply_fill(
        _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    ps.apply_fill(
        _make_report("XYZ", OrderSide.SELL, 100.0, date(2020, 1, 3), 60.0, FlatSlippage(bps=0))
    )
    assert ps.cash == pytest.approx(100_000.0 - 100 * 50.0 + 100 * 60.0)
    assert ps.positions["XYZ"].size == 0.0
    assert ps.positions["XYZ"].realized_pnl == pytest.approx(1000.0)


def test_rejected_fill_is_noop():
    ps = PortfolioState(cash=100_000.0)
    # Below min_size gets rejected.
    order = Order(symbol="XYZ", side=OrderSide.BUY, size=0.5, timestamp=date(2020, 1, 2))
    profile = LiquidityProfile(
        symbol="XYZ",
        adv_series=pd.Series([1e6] * 5, index=pd.date_range("2020-01-01", periods=5)),
        min_size=100.0,
    )
    report = submit_order(order, profile, price=50.0, model=FlatSlippage(bps=0))
    ps.apply_fill(report)
    # Nothing should have moved.
    assert ps.cash == 100_000.0
    assert "XYZ" not in ps.positions or ps.positions["XYZ"].size == 0.0


# ---------- total_value + mark_to_market ------------------------------


def test_total_value_flat_portfolio_is_cash():
    ps = PortfolioState(cash=100_000.0)
    assert ps.total_value(prices={}) == 100_000.0


def test_total_value_with_open_positions():
    ps = PortfolioState(cash=50_000.0)
    ps.apply_fill(
        _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    # Now 100 shares of XYZ at cost 50. Price rises to 60.
    total = ps.total_value(prices={"XYZ": 60.0})
    # cash = 50000 - 5000 = 45000. positions = 100 * 60 = 6000. total = 51000.
    assert total == pytest.approx(45_000.0 + 6_000.0)


def test_total_value_raises_on_missing_symbol_price():
    ps = PortfolioState(cash=50_000.0)
    ps.apply_fill(
        _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    with pytest.raises(KuantValueError):
        ps.total_value(prices={"AAA": 100.0})


def test_total_value_nan_price_propagates():
    """NaN means 'unpriced today', not 'zero-valued'."""
    ps = PortfolioState(cash=50_000.0)
    ps.apply_fill(
        _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    total = ps.total_value(prices={"XYZ": float("nan")})
    assert np.isnan(total)


def test_flat_positions_dont_require_price():
    """A symbol with realized-only P&L (size=0) can be omitted from prices."""
    ps = PortfolioState(cash=100_000.0)
    ps.apply_fill(
        _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    ps.apply_fill(
        _make_report("XYZ", OrderSide.SELL, 100.0, date(2020, 1, 3), 60.0, FlatSlippage(bps=0))
    )
    # XYZ is flat; prices dict is empty.
    total = ps.total_value(prices={})
    assert total == pytest.approx(101_000.0)  # 100k + 1k realized


# ---------- mark_to_market snapshot -----------------------------------


def test_mark_to_market_returns_snapshot():
    ps = PortfolioState(cash=50_000.0)
    ps.apply_fill(
        _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    snap = ps.mark_to_market(prices={"XYZ": 60.0})
    assert isinstance(snap, EquitySnapshot)
    assert snap.cash == pytest.approx(45_000.0)
    assert snap.positions_value == pytest.approx(6_000.0)
    assert snap.total_value == pytest.approx(51_000.0)
    assert snap.n_positions == 1
    assert snap.unrealized_pnl == pytest.approx(1_000.0)
    assert snap.realized_pnl == 0.0


def test_snapshot_realized_pnl_sums_across_symbols():
    ps = PortfolioState(cash=100_000.0)
    ps.apply_fill(
        _make_report("A", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    ps.apply_fill(
        _make_report("A", OrderSide.SELL, 100.0, date(2020, 1, 3), 60.0, FlatSlippage(bps=0))
    )
    ps.apply_fill(
        _make_report("B", OrderSide.BUY, 100.0, date(2020, 1, 2), 20.0, FlatSlippage(bps=0))
    )
    ps.apply_fill(
        _make_report("B", OrderSide.SELL, 100.0, date(2020, 1, 3), 15.0, FlatSlippage(bps=0))
    )
    snap = ps.mark_to_market(prices={})
    # A: +1000, B: -500.
    assert snap.realized_pnl == pytest.approx(500.0)


def test_portfolio_summary_contains_expected_keys():
    ps = PortfolioState(cash=100_000.0)
    ps.apply_fill(
        _make_report("XYZ", OrderSide.BUY, 100.0, date(2020, 1, 2), 50.0, FlatSlippage(bps=0))
    )
    s = ps.summary()
    assert "cash" in s
    assert "n open positions" in s

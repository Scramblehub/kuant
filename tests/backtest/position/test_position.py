"""Tests for kuant.backtest.position.Position."""

from __future__ import annotations

import pytest

from kuant.backtest.position import Position


def test_flat_position_initial_state():
    p = Position(symbol="X")
    assert p.size == 0.0
    assert p.avg_cost == 0.0
    assert p.realized_pnl == 0.0


def test_first_fill_sets_size_and_cost():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)
    assert p.size == 100.0
    assert p.avg_cost == 50.0
    assert p.realized_pnl == 0.0


def test_add_to_long_position_weighted_avg_cost():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)
    p.apply_fill(size_filled=100.0, price=60.0)
    assert p.size == 200.0
    assert p.avg_cost == pytest.approx(55.0)


def test_partial_close_realizes_gain():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)  # long 100 @ 50
    p.apply_fill(size_filled=-40.0, price=70.0)  # sell 40 @ 70 (+20 profit each)
    assert p.size == 60.0
    assert p.avg_cost == pytest.approx(50.0)  # unchanged on remainder
    assert p.realized_pnl == pytest.approx(40 * 20.0)


def test_full_close_flattens():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)
    p.apply_fill(size_filled=-100.0, price=55.0)
    assert p.size == 0.0
    assert p.avg_cost == 0.0
    assert p.realized_pnl == pytest.approx(500.0)


def test_flip_long_to_short_resets_avg_cost_on_remainder():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)  # long 100 @ 50
    p.apply_fill(size_filled=-150.0, price=60.0)  # close 100, then short 50 @ 60
    # Realized on the closed 100 shares: 100 * (60 - 50) = 1000.
    assert p.realized_pnl == pytest.approx(1000.0)
    # Remaining 50 short at avg_cost = 60.
    assert p.size == -50.0
    assert p.avg_cost == pytest.approx(60.0)


def test_flip_short_to_long_resets_avg_cost():
    p = Position(symbol="X")
    p.apply_fill(size_filled=-100.0, price=50.0)  # short 100 @ 50
    p.apply_fill(size_filled=150.0, price=40.0)  # cover 100, then long 50 @ 40
    # Realized on short cover: 100 * (50 - 40) = 1000.
    assert p.realized_pnl == pytest.approx(1000.0)
    assert p.size == 50.0
    assert p.avg_cost == pytest.approx(40.0)


def test_short_position_loss_on_price_rise():
    p = Position(symbol="X")
    p.apply_fill(size_filled=-100.0, price=50.0)  # short 100 @ 50
    p.apply_fill(size_filled=100.0, price=55.0)  # cover 100 @ 55, -500 realized
    assert p.realized_pnl == pytest.approx(-500.0)
    assert p.size == 0.0


def test_zero_fill_is_noop():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)
    p.apply_fill(size_filled=0.0, price=999.0)
    assert p.size == 100.0
    assert p.avg_cost == 50.0


def test_market_value():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)
    assert p.market_value(price=60.0) == pytest.approx(6000.0)


def test_market_value_zero_when_flat():
    p = Position(symbol="X")
    assert p.market_value(price=999.0) == 0.0


def test_unrealized_pnl_long():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)
    assert p.unrealized_pnl(price=60.0) == pytest.approx(1000.0)


def test_unrealized_pnl_short():
    p = Position(symbol="X")
    p.apply_fill(size_filled=-100.0, price=50.0)
    assert p.unrealized_pnl(price=40.0) == pytest.approx(1000.0)


def test_total_pnl_combines_realized_and_unrealized():
    p = Position(symbol="X")
    p.apply_fill(size_filled=100.0, price=50.0)
    p.apply_fill(size_filled=-40.0, price=70.0)  # realized 800
    # Unrealized on remaining 60 shares at, say, price 80: 60 * (80 - 50) = 1800.
    assert p.total_pnl(price=80.0) == pytest.approx(800.0 + 1800.0)


def test_summary_contains_symbol():
    p = Position(symbol="ACME")
    assert "ACME" in p.summary()

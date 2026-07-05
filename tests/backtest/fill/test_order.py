"""Tests for kuant.backtest.fill.Order and enums."""

from __future__ import annotations

from datetime import date

import pytest

from kuant.backtest.fill import Order, OrderSide, OrderStatus, OrderType
from kuant.errors import KuantValueError


def test_order_construct_defaults():
    o = Order(symbol="XYZ", side=OrderSide.BUY, size=100.0, timestamp=date(2020, 1, 2))
    assert o.symbol == "XYZ"
    assert o.side is OrderSide.BUY
    assert o.size == 100.0
    assert o.order_type is OrderType.MARKET
    assert o.limit_price is None
    assert o.order_id > 0
    assert o.tag == ""


def test_signed_size():
    buy = Order(symbol="X", side=OrderSide.BUY, size=100.0, timestamp=date(2020, 1, 2))
    sell = Order(symbol="X", side=OrderSide.SELL, size=100.0, timestamp=date(2020, 1, 2))
    assert buy.signed_size == 100.0
    assert sell.signed_size == -100.0


def test_reject_negative_size():
    with pytest.raises(KuantValueError):
        Order(symbol="X", side=OrderSide.BUY, size=-1.0, timestamp=date(2020, 1, 2))


def test_reject_zero_size():
    with pytest.raises(KuantValueError):
        Order(symbol="X", side=OrderSide.BUY, size=0.0, timestamp=date(2020, 1, 2))


def test_limit_order_requires_price():
    with pytest.raises(KuantValueError):
        Order(
            symbol="X",
            side=OrderSide.BUY,
            size=100.0,
            timestamp=date(2020, 1, 2),
            order_type=OrderType.LIMIT,
        )


def test_limit_order_accepts_price():
    o = Order(
        symbol="X",
        side=OrderSide.BUY,
        size=100.0,
        timestamp=date(2020, 1, 2),
        order_type=OrderType.LIMIT,
        limit_price=50.0,
    )
    assert o.limit_price == 50.0


def test_order_ids_are_monotonic():
    a = Order(symbol="A", side=OrderSide.BUY, size=1.0, timestamp=date(2020, 1, 2))
    b = Order(symbol="B", side=OrderSide.BUY, size=1.0, timestamp=date(2020, 1, 2))
    assert b.order_id > a.order_id


def test_order_summary_contains_symbol():
    o = Order(symbol="ACME", side=OrderSide.BUY, size=100.0, timestamp=date(2020, 1, 2))
    assert "ACME" in o.summary()


def test_order_status_enum_values():
    """The enum's values are stable strings safe for JSON round-trip."""
    assert OrderStatus.PENDING.value == "pending"
    assert OrderStatus.FILLED.value == "filled"
    assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
    assert OrderStatus.REJECTED.value == "rejected"
    assert OrderStatus.CANCELED.value == "canceled"


def test_order_side_enum_values():
    assert OrderSide.BUY.value == 1
    assert OrderSide.SELL.value == -1

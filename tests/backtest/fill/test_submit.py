"""Tests for kuant.backtest.fill.submit_order."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.backtest.fill import (
    FillReport,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    submit_order,
)
from kuant.backtest.liquidity import FlatSlippage, LiquidityProfile
from kuant.errors import KuantValueError


def _profile(
    symbol: str = "XYZ", adv: float = 1_000_000.0, min_size: float = 100.0
) -> LiquidityProfile:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    adv_s = pd.Series([adv] * 5, index=idx, dtype=float)
    return LiquidityProfile(
        symbol=symbol,
        adv_series=adv_s,
        min_size=min_size,
        max_participation=0.10,
    )


def _order(symbol: str = "XYZ", side=OrderSide.BUY, size: float = 1000.0, ts=None):
    return Order(
        symbol=symbol,
        side=side,
        size=size,
        timestamp=ts or date(2020, 1, 2),
    )


def test_submit_market_ok():
    o = _order()
    p = _profile()
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    assert isinstance(report, FillReport)
    assert report.status is OrderStatus.FILLED
    assert report.fill.size_filled == 1000.0
    assert report.symbol == "XYZ"
    assert report.order_id == o.order_id


def test_submit_capped_returns_partially_filled():
    o = _order(size=200_000.0)
    p = _profile()  # cap = 100k
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    assert report.status is OrderStatus.PARTIALLY_FILLED
    assert report.fill.size_filled == 100_000.0


def test_submit_below_min_size_rejected():
    o = _order(size=50.0)
    p = _profile(min_size=100.0)
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    assert report.status is OrderStatus.REJECTED
    assert report.fill.size_filled == 0.0


def test_submit_no_liquidity_rejected():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    adv = pd.Series([1e6, np.nan, 1e6], index=idx)
    p = LiquidityProfile(symbol="XYZ", adv_series=adv, min_size=1.0)
    o = _order(size=1000.0)  # timestamp = 2020-01-02 which has NaN ADV
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    assert report.status is OrderStatus.REJECTED
    assert report.fill.reason == "NO_LIQUIDITY"


def test_submit_missing_date_rejected():
    o = _order(ts=date(2019, 1, 1))
    p = _profile()
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    assert report.status is OrderStatus.REJECTED


def test_submit_rejects_non_market_order():
    """LIMIT / STOP not executable in v1; raise loudly."""
    o = Order(
        symbol="XYZ",
        side=OrderSide.BUY,
        size=100.0,
        timestamp=date(2020, 1, 2),
        order_type=OrderType.LIMIT,
        limit_price=49.0,
    )
    p = _profile()
    with pytest.raises(KuantValueError):
        submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))


def test_submit_rejects_symbol_mismatch():
    o = _order(symbol="AAA")
    p = _profile(symbol="BBB")
    with pytest.raises(KuantValueError):
        submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))


def test_sell_order_produces_negative_signed_size_on_fill():
    o = _order(side=OrderSide.SELL, size=1000.0)
    p = _profile()
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    assert report.fill.size_filled == -1000.0
    # Sell price is lower than reference by the slippage haircut.
    assert report.fill.price < 50.0


def test_report_carries_tag_through():
    o = Order(
        symbol="XYZ",
        side=OrderSide.BUY,
        size=1000.0,
        timestamp=date(2020, 1, 2),
        tag="momentum_v3",
    )
    p = _profile()
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    assert report.tag == "momentum_v3"


def test_report_summary_contains_symbol_and_status():
    o = _order()
    p = _profile()
    report = submit_order(o, p, price=50.0, model=FlatSlippage(bps=5))
    s = report.summary()
    assert "XYZ" in s
    assert "filled" in s

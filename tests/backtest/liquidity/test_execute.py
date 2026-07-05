"""Tests for execute_fill / execute_fill_panel / liquidity_mask."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.backtest.liquidity import (
    FlatSlippage,
    LinearImpact,
    LiquidityProfile,
    execute_fill,
    execute_fill_panel,
    liquidity_mask,
)
from kuant.errors import KuantShapeError, KuantValueError


def _profile(
    n: int = 5, adv: float = 1_000_000.0, min_size: float = 100.0, max_participation: float = 0.10
) -> LiquidityProfile:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    adv_s = pd.Series([adv] * n, index=idx, dtype=float)
    return LiquidityProfile(
        symbol="XYZ",
        adv_series=adv_s,
        min_size=min_size,
        max_participation=max_participation,
    )


# ---------- execute_fill (single) --------------------------------------


def test_fill_ok_within_cap():
    p = _profile()
    r = execute_fill(1000.0, 50.0, p, date(2020, 1, 2), FlatSlippage(bps=5))
    assert r.reason == "OK"
    assert r.size_filled == 1000.0
    assert r.size_rejected == 0.0
    assert r.price == pytest.approx(50.0 * 1.0005)
    assert r.slippage_bps == pytest.approx(5.0)


def test_fill_capped_at_participation():
    p = _profile(adv=1_000_000.0, max_participation=0.1)
    # Request 200k shares → cap is 100k.
    r = execute_fill(200_000.0, 50.0, p, date(2020, 1, 2), FlatSlippage(bps=5))
    assert r.reason == "CAPPED_PARTICIPATION"
    assert r.size_filled == 100_000.0
    assert r.size_rejected == 100_000.0


def test_fill_below_min_size_rejected():
    p = _profile(min_size=100.0)
    r = execute_fill(50.0, 50.0, p, date(2020, 1, 2), FlatSlippage(bps=5))
    assert r.reason == "BELOW_MIN_SIZE"
    assert r.size_filled == 0.0
    assert r.size_rejected == 50.0


def test_fill_no_liquidity_when_adv_nan():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    adv = pd.Series([1e6, np.nan, 1e6], index=idx)
    p = LiquidityProfile(symbol="X", adv_series=adv, min_size=1.0)
    r = execute_fill(1000.0, 50.0, p, date(2020, 1, 2), FlatSlippage(bps=5))
    assert r.reason == "NO_LIQUIDITY"


def test_fill_missing_date():
    p = _profile()
    r = execute_fill(1000.0, 50.0, p, date(2019, 1, 1), FlatSlippage(bps=5))
    assert r.reason == "MISSING_DATE"


def test_fill_sell_side_symmetric_price():
    p = _profile()
    r_buy = execute_fill(1000.0, 50.0, p, date(2020, 1, 2), FlatSlippage(bps=10))
    r_sell = execute_fill(-1000.0, 50.0, p, date(2020, 1, 2), FlatSlippage(bps=10))
    # Buy pays more, sell receives less.
    assert r_buy.price > 50.0
    assert r_sell.price < 50.0
    assert (r_buy.price - 50.0) == pytest.approx(50.0 - r_sell.price)


def test_fill_linear_impact_scales_with_order_size():
    p = _profile(adv=1_000_000.0)
    small = execute_fill(50_000.0, 50.0, p, date(2020, 1, 2), LinearImpact(k=20.0))
    # 200_000 also fits under 10% ADV cap? No — 20% > 10%. Use 100k which is at cap.
    at_cap = execute_fill(100_000.0, 50.0, p, date(2020, 1, 2), LinearImpact(k=20.0))
    # 5% participation → 1 bp. 10% participation → 2 bp.
    assert small.slippage_bps == pytest.approx(1.0)
    assert at_cap.slippage_bps == pytest.approx(2.0)


def test_fill_rejects_non_positive_price():
    p = _profile()
    with pytest.raises(KuantValueError):
        execute_fill(1000.0, 0.0, p, date(2020, 1, 2), FlatSlippage(bps=5))


def test_fill_rejects_model_without_compute_slippage():
    p = _profile()
    with pytest.raises(KuantValueError):
        execute_fill(1000.0, 50.0, p, date(2020, 1, 2), object())


def test_fill_result_summary_contains_reason():
    p = _profile()
    r = execute_fill(1000.0, 50.0, p, date(2020, 1, 2), FlatSlippage(bps=5))
    assert "OK" in r.summary()


# ---------- execute_fill_panel ----------------------------------------


def test_fill_panel_returns_dataframe():
    p = _profile()
    orders = pd.DataFrame(
        {
            "timestamp": [date(2020, 1, 2), date(2020, 1, 3)],
            "size": [1000.0, -500.0],
            "price": [50.0, 51.0],
        }
    )
    out = execute_fill_panel(orders, p, FlatSlippage(bps=5))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "timestamp",
        "size",
        "price",
        "fill_price",
        "size_filled",
        "size_rejected",
        "slippage_bps",
        "reason",
        "cost",
    ]
    assert len(out) == 2
    assert out["reason"].tolist() == ["OK", "OK"]


def test_fill_panel_reject_non_dataframe():
    with pytest.raises(KuantShapeError):
        execute_fill_panel(np.zeros((3, 3)), _profile(), FlatSlippage(bps=1))


def test_fill_panel_reject_missing_columns():
    orders = pd.DataFrame({"timestamp": [date(2020, 1, 2)], "size": [100.0]})
    with pytest.raises(KuantValueError):
        execute_fill_panel(orders, _profile(), FlatSlippage(bps=1))


def test_fill_panel_mixed_reasons():
    """Panel with a rejected order alongside filled orders reports both."""
    p = _profile(min_size=100.0)
    orders = pd.DataFrame(
        {
            "timestamp": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)],
            "size": [1000.0, 50.0, 200_000.0],  # OK, below-min, over-cap
            "price": [50.0, 50.0, 50.0],
        }
    )
    out = execute_fill_panel(orders, p, FlatSlippage(bps=5))
    assert out["reason"].tolist() == [
        "OK",
        "BELOW_MIN_SIZE",
        "CAPPED_PARTICIPATION",
    ]


# ---------- liquidity_mask --------------------------------------------


def test_liquidity_mask_masks_nan_adv():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    adv = pd.Series([1e6, np.nan, 1e6, 1e6], index=idx)
    p = LiquidityProfile(symbol="X", adv_series=adv, min_size=1.0)
    m = liquidity_mask(idx, p)
    assert m.tolist() == [True, False, True, True]


def test_liquidity_mask_masks_below_threshold():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    adv = pd.Series([100.0, 200.0, 5000.0, 10000.0], index=idx)
    p = LiquidityProfile(symbol="X", adv_series=adv, min_size=1.0)
    m = liquidity_mask(idx, p, min_adv=1000.0)
    assert m.tolist() == [False, False, True, True]


def test_liquidity_mask_composes_with_lifecycle_mask():
    """The whole point of exposing liquidity_mask as a separate primitive."""
    from kuant.backtest.lifecycle import (
        SecurityLifecycle,
        tradeable_mask,
    )

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    adv = pd.Series([1e6, 1e6, 0.0, 1e6, 1e6], index=idx)
    p = LiquidityProfile(symbol="X", adv_series=adv)
    lc = SecurityLifecycle(symbol="X", delisting_date=date(2020, 1, 4))

    lm = liquidity_mask(idx, p)
    tm = tradeable_mask(idx, lc)
    combined = lm & tm
    # Day 1,2,4 fine. Day 3 → no liquidity. Day 5 → after delisting.
    assert combined.tolist() == [True, True, False, True, False]

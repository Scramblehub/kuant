"""Tests for FillModel implementations."""

from __future__ import annotations

import math

import pytest

from kuant.backtest.liquidity import FlatSlippage, LinearImpact, SquareRootImpact
from kuant.errors import KuantValueError


# ---------- FlatSlippage ------------------------------------------------


def test_flat_slippage_constant():
    m = FlatSlippage(bps=10.0)
    assert m.compute_slippage(1000.0, 1_000_000.0, 1) == pytest.approx(0.001)


def test_flat_slippage_zero_bps_is_zero():
    m = FlatSlippage(bps=0.0)
    assert m.compute_slippage(100.0, 1e6, 1) == 0.0


def test_flat_slippage_ignores_size():
    m = FlatSlippage(bps=25.0)
    small = m.compute_slippage(100.0, 1e6, 1)
    large = m.compute_slippage(1e6, 1e6, 1)
    assert small == large == pytest.approx(0.0025)


def test_flat_slippage_reject_negative_bps():
    with pytest.raises(KuantValueError):
        FlatSlippage(bps=-1.0)


# ---------- LinearImpact ------------------------------------------------


def test_linear_impact_scales_with_participation():
    m = LinearImpact(k=20.0)
    at_10pct = m.compute_slippage(100_000.0, 1_000_000.0, 1)
    at_5pct = m.compute_slippage(50_000.0, 1_000_000.0, 1)
    assert at_10pct == pytest.approx(2e-4)  # 20 bps * 0.1 = 2 bps
    assert at_5pct == pytest.approx(1e-4)  # 20 bps * 0.05 = 1 bp


def test_linear_impact_symmetric_in_side():
    m = LinearImpact(k=20.0)
    buy = m.compute_slippage(100_000.0, 1_000_000.0, 1)
    sell = m.compute_slippage(100_000.0, 1_000_000.0, -1)
    assert buy == sell


def test_linear_impact_zero_adv_raises():
    m = LinearImpact(k=20.0)
    with pytest.raises(KuantValueError):
        m.compute_slippage(100.0, 0.0, 1)


def test_linear_impact_reject_negative_k():
    with pytest.raises(KuantValueError):
        LinearImpact(k=-1.0)


# ---------- SquareRootImpact --------------------------------------------


def test_square_root_impact_scales_correctly():
    m = SquareRootImpact(k=20.0)
    at_100pct = m.compute_slippage(1_000_000.0, 1_000_000.0, 1)
    at_25pct = m.compute_slippage(250_000.0, 1_000_000.0, 1)
    assert at_100pct == pytest.approx(20e-4)
    # 25% participation → sqrt(0.25) = 0.5 → half the slippage of 100%.
    assert at_25pct == pytest.approx(10e-4)


def test_square_root_is_concave_in_size():
    """Twice the size pays less than twice the slippage."""
    m = SquareRootImpact(k=30.0)
    small = m.compute_slippage(100_000.0, 1_000_000.0, 1)
    double = m.compute_slippage(200_000.0, 1_000_000.0, 1)
    # sqrt(2)*small ~= 1.414*small < 2*small
    assert double < 2 * small
    assert double == pytest.approx(math.sqrt(2) * small, rel=1e-9)


def test_square_root_impact_zero_adv_raises():
    m = SquareRootImpact(k=20.0)
    with pytest.raises(KuantValueError):
        m.compute_slippage(100.0, 0.0, 1)


def test_square_root_impact_reject_negative_k():
    with pytest.raises(KuantValueError):
        SquareRootImpact(k=-1.0)

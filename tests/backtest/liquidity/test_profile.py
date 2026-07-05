"""Tests for LiquidityProfile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kuant.backtest.liquidity import LiquidityProfile
from kuant.errors import KuantShapeError, KuantValueError


def _adv(n: int = 5) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.linspace(1e6, 1.1e6, n), index=idx)


def test_construct_defaults():
    p = LiquidityProfile(symbol="XYZ", adv_series=_adv())
    assert p.symbol == "XYZ"
    assert p.spread_series is None
    assert p.min_size == 1.0
    assert p.max_participation == 0.10


def test_construct_with_spread():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    adv = pd.Series([1e6, 1e6, 1e6], index=idx)
    spread = pd.Series([2.0, 3.0, 2.5], index=idx)
    p = LiquidityProfile(symbol="XYZ", adv_series=adv, spread_series=spread)
    assert p.spread_series is spread


def test_reject_non_series_adv():
    with pytest.raises(KuantShapeError):
        LiquidityProfile(symbol="X", adv_series=np.arange(5.0))


def test_reject_spread_length_mismatch():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    adv = pd.Series([1e6, 1e6, 1e6], index=idx)
    idx2 = pd.date_range("2020-01-01", periods=4, freq="D")
    spread = pd.Series([2.0, 3.0, 2.5, 2.0], index=idx2)
    with pytest.raises(KuantShapeError):
        LiquidityProfile(symbol="X", adv_series=adv, spread_series=spread)


def test_reject_nonpositive_min_size():
    with pytest.raises(KuantValueError):
        LiquidityProfile(symbol="X", adv_series=_adv(), min_size=0.0)


def test_reject_max_participation_out_of_range():
    with pytest.raises(KuantValueError):
        LiquidityProfile(symbol="X", adv_series=_adv(), max_participation=1.5)


def test_reject_max_participation_zero():
    with pytest.raises(KuantValueError):
        LiquidityProfile(symbol="X", adv_series=_adv(), max_participation=0.0)


def test_summary_contains_symbol():
    p = LiquidityProfile(symbol="ACME", adv_series=_adv())
    s = p.summary()
    assert "ACME" in s

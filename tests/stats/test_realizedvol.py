"""Tests for kuant.stats.realizedvol."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.stats.realizedvol import atr, garmanklass, parkinson, rogerssatchell, yangzhang


def _synthetic_ohlc(n=500, seed=0):
    """Log-normal walk so prices stay strictly positive on long runs."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.002, n)))
    return open_, high, low, close


# ---------- atr --------------------------------------------------------


def test_atr_shape_and_warmup():
    _, H, L, C = _synthetic_ohlc()
    a = atr(H, L, C, window=14)
    assert a.shape == H.shape
    assert np.isnan(a[:13]).all()
    assert np.isfinite(a[13:]).any()


def test_atr_positive_when_range_positive():
    """ATR is nonnegative by construction."""
    _, H, L, C = _synthetic_ohlc()
    a = atr(H, L, C, window=14)
    finite = a[np.isfinite(a)]
    assert (finite >= 0).all()


def test_atr_scales_with_range():
    """Doubling ONLY the H-L width (holding H and L symmetric around close)
    roughly doubles the ATR. Keep close fixed so the |H - prev_C| terms
    don't dominate the max in the True Range formula."""
    n = 200
    close = np.full(n, 100.0)
    half = np.full(n, 1.0)
    H = close + half
    L = close - half
    a1 = atr(H, L, close, window=14)
    a2 = atr(close + 2 * half, close - 2 * half, close, window=14)
    ratio = np.nanmean(a2) / np.nanmean(a1)
    assert abs(ratio - 2.0) < 0.05


def test_atr_rejects_length_mismatch():
    with pytest.raises(Exception):
        atr(np.arange(10.0), np.arange(9.0), np.arange(10.0))


# ---------- parkinson / garmanklass / rogerssatchell -------------------


def test_parkinson_positive():
    _, H, L, _ = _synthetic_ohlc()
    assert parkinson(H, L) > 0.0


def test_parkinson_rejects_non_positive():
    with pytest.raises(KuantValueError):
        parkinson(np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))


def test_garmanklass_positive():
    O_, H, L, C = _synthetic_ohlc()
    assert garmanklass(O_, H, L, C) > 0.0


def test_rogerssatchell_positive():
    O_, H, L, C = _synthetic_ohlc()
    assert rogerssatchell(O_, H, L, C) > 0.0


def test_estimators_agree_within_order_of_magnitude():
    """Parkinson, GK, RS should all be in the same ballpark for GBM-like data."""
    O_, H, L, C = _synthetic_ohlc(n=2000, seed=42)
    p = parkinson(H, L)
    g = garmanklass(O_, H, L, C)
    r = rogerssatchell(O_, H, L, C)
    # Within 2x is a reasonable check.
    for x in (g, r):
        assert 0.3 < x / p < 3.0


# ---------- yang-zhang -------------------------------------------------


def test_yangzhang_positive():
    O_, H, L, C = _synthetic_ohlc(n=2000)
    assert yangzhang(O_, H, L, C) > 0.0


def test_yangzhang_rejects_too_short():
    with pytest.raises(KuantValueError):
        yangzhang(np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([100.0]))


def test_yangzhang_with_explicit_prev_close():
    O_, H, L, C = _synthetic_ohlc(n=500)
    pc = C - 0.5
    v = yangzhang(O_, H, L, C, prev_close=pc)
    assert np.isfinite(v)
    assert v > 0

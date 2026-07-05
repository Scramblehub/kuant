"""Tests for kuant.portfolio.sharperatio."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantNumericWarning, KuantValueError
from kuant.portfolio.sharperatio import SharpeResult, sharperatio


# ---------- known-truth --------------------------------------------------


def test_sharpe_matches_manual_formula():
    """Sharpe = mean * sqrt(ann) / std under zero risk-free rate."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    result = sharperatio(r, ann_factor=252)
    expected = float(np.mean(r) * np.sqrt(252) / np.std(r, ddof=1))
    assert abs(result.sharpe - expected) < 1e-9


def test_positive_expected_return_positive_sharpe():
    rng = np.random.default_rng(0)
    r = rng.normal(0.005, 0.01, 500)  # positive drift
    result = sharperatio(r)
    assert result.sharpe > 0.0


def test_negative_expected_return_negative_sharpe():
    rng = np.random.default_rng(0)
    r = rng.normal(-0.005, 0.01, 500)
    result = sharperatio(r)
    assert result.sharpe < 0.0


def test_risk_free_subtraction():
    """Subtracting a per-period rf that equals the mean returns Sharpe = 0."""
    rng = np.random.default_rng(0)
    mean = 0.002
    r = rng.normal(mean, 0.01, 1000)
    # rf = sample mean → excess mean ≈ 0 → sharpe ≈ 0
    result = sharperatio(r, rf=float(np.mean(r)))
    assert abs(result.sharpe) < 0.01


# ---------- annualization scaling ---------------------------------------


def test_higher_ann_factor_scales_up():
    """Sharpe scales with sqrt(ann_factor)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    s_daily = sharperatio(r, ann_factor=252).sharpe
    s_annual = sharperatio(r, ann_factor=1).sharpe
    assert abs(s_daily / s_annual - np.sqrt(252)) < 1e-9


# ---------- NaN + edge cases --------------------------------------------


def test_nan_dropped_from_computation():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 200)
    r_with_nan = r.copy()
    r_with_nan[50:70] = np.nan
    result = sharperatio(r_with_nan)
    assert result.n == 180


def test_constant_returns_sharpe_zero():
    """Constant returns → std at FP-noise level → Sharpe pinned to 0."""
    result = sharperatio(np.full(100, 0.01))
    assert result.sharpe == 0.0
    # std comes back at floating-point noise level, not exactly 0.
    assert result.std < 1e-15


def test_reject_all_nan():
    with pytest.raises(KuantValueError):
        sharperatio(np.array([np.nan, np.nan]))


# ---------- warnings ----------------------------------------------------


def test_small_sample_warns():
    """Under 30 finite observations → warning."""
    with pytest.warns(KuantNumericWarning) as record:
        sharperatio(np.arange(20, dtype=np.float64))
    assert any("KW-SHARPE-SMALL-SAMPLE" in str(w.message) for w in record)


def test_large_sample_no_warning():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 100)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", KuantNumericWarning)
        sharperatio(r)  # would raise if a warning fired


# ---------- error contract ----------------------------------------------


def test_reject_2d_input():
    with pytest.raises(Exception):
        sharperatio(np.zeros((10, 3)))


def test_reject_negative_ann_factor():
    with pytest.raises(KuantValueError):
        sharperatio(np.arange(50.0), ann_factor=-1)


# ---------- result contract ---------------------------------------------


def test_returns_dataclass():
    r = sharperatio(np.arange(50.0))
    assert isinstance(r, SharpeResult)


def test_summary_contains_metadata():
    r = sharperatio(np.arange(100.0))
    s = r.summary()
    assert "SharpeResult" in s
    assert "annualized Sharpe" in s

"""Tests for kuant.portfolio.sortinoratio."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantNumericWarning, KuantValueError
from kuant.portfolio.sortinoratio import SortinoResult, sortinoratio


# ---------- known-truth --------------------------------------------------


def test_sortino_matches_manual_formula():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    result = sortinoratio(r, ann_factor=252)
    excess = r - 0.0
    downside = np.minimum(excess, 0.0)
    expected = float(np.mean(excess) * np.sqrt(252) / np.sqrt(np.mean(downside**2)))
    assert abs(result.sortino - expected) < 1e-9


def test_sortino_positive_on_positive_drift():
    rng = np.random.default_rng(0)
    r = rng.normal(0.005, 0.01, 1000)
    assert sortinoratio(r).sortino > 0.0


def test_sortino_negative_on_negative_drift():
    rng = np.random.default_rng(0)
    r = rng.normal(-0.005, 0.01, 1000)
    assert sortinoratio(r).sortino < 0.0


# ---------- Sharpe/Sortino relationship ---------------------------------


def test_symmetric_returns_sortino_close_to_sharpe_scaled():
    """On symmetric returns, downside variance ≈ full variance / 2, so
    sortino ≈ sharpe * sqrt(2). Not exact on any finite sample."""
    from kuant.portfolio import sharperatio

    rng = np.random.default_rng(42)
    r = rng.normal(0.001, 0.01, 5000)  # symmetric
    sortino = sortinoratio(r, ann_factor=252).sortino
    sharpe = sharperatio(r, ann_factor=252).sharpe
    ratio = sortino / sharpe
    # Should be in the ballpark of sqrt(2) ≈ 1.414.
    assert 1.2 < ratio < 1.6


def test_upside_only_returns_infinite_sortino():
    """No downside excursions → Sortino diverges."""
    r = np.array([0.01, 0.02, 0.03] * 20)  # all positive, n=60 to avoid small-sample warn
    with pytest.warns(KuantNumericWarning) as record:
        result = sortinoratio(r)
    assert any("KW-SORTINO-NO-DOWNSIDE" in str(w.message) for w in record)
    assert result.sortino == float("inf")


# ---------- target parameter --------------------------------------------


def test_target_shifts_downside_boundary():
    """Raising target should make more observations 'below', increasing
    downside variance."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    lo = sortinoratio(r, target=-0.05)
    hi = sortinoratio(r, target=0.005)
    assert hi.n_below_target > lo.n_below_target


# ---------- NaN + edge cases --------------------------------------------


def test_nan_dropped():
    r = np.random.default_rng(0).normal(0.001, 0.01, 200)
    r_with_nan = r.copy()
    r_with_nan[50:70] = np.nan
    result = sortinoratio(r_with_nan)
    assert result.n == 180


def test_reject_all_nan():
    with pytest.raises(KuantValueError):
        sortinoratio(np.array([np.nan, np.nan]))


# ---------- warnings ----------------------------------------------------


def test_small_sample_warns():
    with pytest.warns(KuantNumericWarning) as record:
        # Include at least one below-target so we don't also trip the
        # NO-DOWNSIDE warning.
        r = np.concatenate([np.full(15, 0.01), np.full(10, -0.005)])
        sortinoratio(r)
    assert any("KW-SORTINO-SMALL-SAMPLE" in str(w.message) for w in record)


# ---------- error contract ----------------------------------------------


def test_reject_negative_ann_factor():
    with pytest.raises(KuantValueError):
        sortinoratio(np.arange(50.0), ann_factor=-1)


def test_reject_2d_input():
    with pytest.raises(Exception):
        sortinoratio(np.zeros((10, 3)))


# ---------- result contract ---------------------------------------------


def test_returns_dataclass():
    r = sortinoratio(np.arange(-50.0, 50.0))
    assert isinstance(r, SortinoResult)


def test_summary_contains_metadata():
    r = sortinoratio(np.arange(-50.0, 50.0))
    s = r.summary()
    assert "SortinoResult" in s
    assert "downside" in s.lower()


def test_result_carries_target():
    r = sortinoratio(np.arange(-50.0, 50.0), target=0.5)
    assert r.target == 0.5

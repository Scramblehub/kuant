"""Tests for kuant.sindy.chaos.embedding."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.embedding import (
    FalseNearestResult,
    MutualInfoResult,
    _embed,
    falsenearest,
    mutualinfo,
)


class TestEmbedHelper:
    def test_embed_shape(self):
        x = np.arange(20.0)
        E = _embed(x, m=3, tau=2)
        assert E.shape == (20 - 2 * 2, 3)

    def test_embed_values(self):
        x = np.arange(10.0)
        E = _embed(x, m=3, tau=1)
        # Row 0: [0, 1, 2]; row 1: [1, 2, 3]; ...
        assert E[0].tolist() == [0.0, 1.0, 2.0]
        assert E[1].tolist() == [1.0, 2.0, 3.0]
        assert E.shape == (8, 3)


class TestMutualInfoAuto:
    def test_returns_result_dataclass(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        r = mutualinfo(x, max_lag=16)
        assert isinstance(r, MutualInfoResult)
        assert r.lags.size == 16
        assert r.mi.size == 16

    def test_mi_nonnegative(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=500)
        r = mutualinfo(x, max_lag=8)
        # Histogram MI is non-negative up to binning noise.
        assert (r.mi >= -1e-8).all()

    def test_suggested_tau_positive(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=800)
        r = mutualinfo(x, max_lag=20)
        assert r.suggested_tau >= 1

    def test_periodic_signal_has_first_min_at_quarter_period(self):
        # sin(2 pi k / 20) has first auto-MI minimum near a quarter period.
        n = 800
        x = np.sin(2 * np.pi * np.arange(n) / 20.0)
        r = mutualinfo(x, max_lag=30)
        assert 3 <= r.suggested_tau <= 8

    def test_summary(self):
        rng = np.random.default_rng(3)
        r = mutualinfo(rng.normal(size=300), max_lag=8)
        s = r.summary()
        assert "MutualInfoResult" in s
        assert "suggested tau" in s


class TestMutualInfoCross:
    def test_cross_mi_scalar(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=500)
        y = 0.9 * x + 0.1 * rng.normal(size=500)
        val = mutualinfo(x, y, lag=1)
        assert isinstance(val, float)
        assert val > 0

    def test_cross_mi_equal_length_required(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=500)
        y = rng.normal(size=400)
        with pytest.raises(KuantValueError, match=r"same length"):
            mutualinfo(x, y, lag=1)


class TestMutualInfoValidation:
    def test_2d_x_rejected(self):
        with pytest.raises(KuantValueError):
            mutualinfo(np.zeros((10, 3)))

    def test_too_few_finite_rejected(self):
        with pytest.raises(KuantValueError, match=r"finite"):
            mutualinfo(np.array([1.0, 2.0, 3.0]))

    def test_bad_bins_rejected(self):
        rng = np.random.default_rng(6)
        with pytest.raises(KuantValueError):
            mutualinfo(rng.normal(size=200), bins=0)

    def test_lag_too_large_rejected(self):
        rng = np.random.default_rng(7)
        with pytest.raises(KuantValueError):
            mutualinfo(rng.normal(size=300), max_lag=1000)


class TestFalseNearest:
    def test_returns_result(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=500)
        r = falsenearest(x, tau=1, max_dim=6)
        assert isinstance(r, FalseNearestResult)
        assert r.dims.size == 6
        assert r.fnn.shape == r.dims.shape

    def test_fnn_fraction_in_unit(self):
        rng = np.random.default_rng(9)
        r = falsenearest(rng.normal(size=500), tau=1, max_dim=5)
        assert (r.fnn >= 0).all() and (r.fnn <= 1).all()

    def test_suggested_m_bounded(self):
        rng = np.random.default_rng(10)
        r = falsenearest(rng.normal(size=500), tau=1, max_dim=5)
        assert 1 <= r.suggested_m <= 5

    def test_2d_x_rejected(self):
        with pytest.raises(KuantValueError):
            falsenearest(np.zeros((10, 3)), tau=1)

    def test_too_few_rejected(self):
        with pytest.raises(KuantValueError):
            falsenearest(np.arange(50.0), tau=1)

    def test_summary(self):
        rng = np.random.default_rng(11)
        r = falsenearest(rng.normal(size=300), tau=1, max_dim=3)
        s = r.summary()
        assert "FalseNearestResult" in s

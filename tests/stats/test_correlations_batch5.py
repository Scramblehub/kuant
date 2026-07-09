"""Tests for kuant.stats v0.6.0 batch 5: correlation variants."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.stats import (
    chatterjeexi,
    distancecorr,
    downsidecorr,
    kendalltau,
    spearmanrank,
)


def _rng():
    return np.random.default_rng(0)


# ---------- kendalltau ------------------------------------------------


class TestKendallTau:
    def test_returns_result(self):
        rng = _rng()
        r = kendalltau(rng.normal(size=200), rng.normal(size=200))
        assert hasattr(r, "coef")
        assert hasattr(r, "p_value")

    def test_perfect_positive_gives_one(self):
        x = np.arange(100.0)
        r = kendalltau(x, x)
        assert r.coef > 0.99

    def test_perfect_negative_gives_minus_one(self):
        x = np.arange(100.0)
        r = kendalltau(x, -x)
        assert r.coef < -0.99

    def test_independent_p_high(self):
        rng = np.random.default_rng(1)
        r = kendalltau(rng.normal(size=500), rng.normal(size=500))
        assert r.p_value > 0.05

    def test_unequal_length_rejected(self):
        with pytest.raises(KuantValueError):
            kendalltau(np.arange(100.0), np.arange(50.0))


# ---------- spearmanrank ---------------------------------------------


class TestSpearman:
    def test_returns_result(self):
        rng = _rng()
        r = spearmanrank(rng.normal(size=200), rng.normal(size=200))
        assert hasattr(r, "coef")

    def test_monotone_transform_invariant(self):
        # Spearman = 1 for any monotone transform of x.
        rng = np.random.default_rng(2)
        x = rng.normal(size=200)
        r = spearmanrank(x, np.exp(x))
        assert r.coef > 0.99

    def test_independent_p_high(self):
        rng = np.random.default_rng(3)
        r = spearmanrank(rng.normal(size=500), rng.normal(size=500))
        assert r.p_value > 0.05


# ---------- distancecorr ---------------------------------------------


class TestDistanceCorr:
    def test_returns_result(self):
        rng = _rng()
        r = distancecorr(rng.normal(size=200), rng.normal(size=200))
        assert hasattr(r, "coef")
        assert 0 <= r.coef <= 1

    def test_independent_near_zero(self):
        rng = np.random.default_rng(4)
        r = distancecorr(rng.normal(size=500), rng.normal(size=500))
        # Distance corr has positive bias; still small for independent.
        assert r.coef < 0.15

    def test_catches_nonlinear_dependence(self):
        # Pearson gives ~0 for y = x^2; distance corr should catch it.
        rng = np.random.default_rng(5)
        x = rng.normal(size=500)
        y = x**2 + 0.3 * rng.normal(size=500)
        r = distancecorr(x, y)
        assert r.coef > 0.3  # strong non-Pearson signal

    def test_perfect_dependence_high(self):
        x = np.arange(300.0)
        r = distancecorr(x, x)
        assert r.coef > 0.99


# ---------- chatterjeexi ---------------------------------------------


class TestChatterjeeXi:
    def test_returns_result(self):
        rng = _rng()
        r = chatterjeexi(rng.normal(size=200), rng.normal(size=200))
        assert hasattr(r, "coef")
        assert r.coef <= 1.0

    def test_functional_dependence_near_one(self):
        # y = f(x) with any measurable f -> xi -> 1 as n grows.
        x = np.linspace(0, 4 * np.pi, 1000)
        y = np.sin(x)
        r = chatterjeexi(x, y)
        assert r.coef > 0.7  # asymptotically 1

    def test_independent_near_zero(self):
        rng = np.random.default_rng(6)
        r = chatterjeexi(rng.normal(size=1000), rng.normal(size=1000))
        assert abs(r.coef) < 0.15

    def test_p_value_valid(self):
        rng = np.random.default_rng(7)
        r = chatterjeexi(rng.normal(size=500), rng.normal(size=500))
        assert 0.0 <= r.p_value <= 1.0


# ---------- downsidecorr ---------------------------------------------


class TestDownsideCorr:
    def test_returns_result(self):
        rng = _rng()
        x = rng.normal(size=500)
        y = 0.7 * x + 0.5 * rng.normal(size=500)
        r = downsidecorr(x, y)
        assert hasattr(r, "coef")

    def test_positive_for_coupled(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=500)
        y = 0.8 * x + 0.3 * rng.normal(size=500)
        r = downsidecorr(x, y)
        assert r.coef > 0.2

    def test_threshold_zero_default(self):
        # If threshold is default (0), n_down should be ~ n/4 for
        # standard normals (roughly p(both < 0) = 0.5 * 0.5 = 0.25).
        rng = np.random.default_rng(9)
        r = downsidecorr(rng.normal(size=1000), rng.normal(size=1000))
        assert 150 < r.n < 350

    def test_returns_nan_if_no_downside(self):
        # All positive series never meet the threshold.
        x = np.abs(np.random.default_rng(10).normal(size=200)) + 1
        y = np.abs(np.random.default_rng(11).normal(size=200)) + 1
        r = downsidecorr(x, y)
        assert np.isnan(r.coef)

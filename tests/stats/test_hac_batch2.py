"""Tests for kuant.stats v0.6.0 batch 2: HAC + autocorr + normality + BDS + spectralentropy."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError, KuantShapeError
from kuant.stats import (
    andersondarling,
    andrewsse,
    bdstest,
    boxpierce,
    durbinwatson,
    jarquebera,
    ljungbox,
    neweywestse,
    shapirowilk,
    spectralentropy,
)


# ---------- HAC (Newey-West + Andrews) --------------------------------


class TestHac:
    def _make(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        X[:, 0] = 1.0
        y = X @ np.array([0.5, 0.3]) + rng.normal(size=n)
        return y, X

    def test_neweywest_positive_se(self):
        y, X = self._make()
        r = neweywestse(y, X)
        assert (r.se > 0).all()
        # Betas close to true (0.5, 0.3).
        assert abs(r.beta[0] - 0.5) < 0.2
        assert abs(r.beta[1] - 0.3) < 0.2

    def test_andrews_positive_se(self):
        y, X = self._make()
        r = andrewsse(y, X)
        assert (r.se > 0).all()

    def test_bad_shape_rejected(self):
        y = np.random.default_rng(0).normal(size=100)
        X = np.random.default_rng(0).normal(size=(80, 2))
        with pytest.raises(KuantShapeError):
            neweywestse(y, X)

    def test_summary(self):
        y, X = self._make()
        r = neweywestse(y, X)
        assert "HacResult" in r.summary()


# ---------- Ljung-Box / Box-Pierce / Durbin-Watson --------------------


class TestAutocorrTests:
    def test_ljungbox_iid_not_significant(self):
        rng = np.random.default_rng(0)
        r = ljungbox(rng.normal(size=500), h=10)
        assert r.p_value > 0.05  # noise shouldn't reject H0 of iid

    def test_ljungbox_ar1_significant(self):
        rng = np.random.default_rng(1)
        n = 500
        x = np.zeros(n)
        x[0] = rng.normal()
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + rng.normal()
        r = ljungbox(x, h=10)
        assert r.p_value < 0.01

    def test_boxpierce_matches_shape(self):
        rng = np.random.default_rng(2)
        r = boxpierce(rng.normal(size=300), h=10)
        assert r.stat > 0

    def test_durbinwatson_iid_near_two(self):
        rng = np.random.default_rng(3)
        r = durbinwatson(rng.normal(size=500))
        assert 1.7 < r.stat < 2.3

    def test_durbinwatson_ar1_low(self):
        rng = np.random.default_rng(4)
        n = 500
        x = np.zeros(n)
        x[0] = rng.normal()
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + rng.normal()
        r = durbinwatson(x)
        assert r.stat < 1.0  # positive autocorrelation

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            ljungbox(np.arange(10.0))


# ---------- Normality tests -------------------------------------------


class TestNormalityTests:
    def test_jarquebera_gaussian_not_reject(self):
        rng = np.random.default_rng(0)
        r = jarquebera(rng.normal(size=1000))
        assert r.p_value > 0.05

    def test_jarquebera_uniform_reject(self):
        rng = np.random.default_rng(1)
        r = jarquebera(rng.uniform(-1, 1, 500))
        # Uniform has kurt = 1.8 vs 3 for normal; JB should reject.
        assert r.p_value < 0.05

    def test_andersondarling_gaussian_p_high(self):
        rng = np.random.default_rng(2)
        r = andersondarling(rng.normal(size=500))
        assert r.p_value > 0.05

    def test_shapirowilk_gaussian_p_high(self):
        rng = np.random.default_rng(3)
        r = shapirowilk(rng.normal(size=200))
        assert r.p_value > 0.05


# ---------- BDS -------------------------------------------------------


class TestBds:
    def test_iid_stat_near_zero(self):
        rng = np.random.default_rng(0)
        r = bdstest(rng.normal(size=800), m=2)
        # Under H0 (iid), stat ~ N(0, 1).
        assert abs(r.stat) < 3.0

    def test_logistic_map_stat_elevated(self):
        # r=4 logistic map is deterministic (not iid); BDS should flag.
        # Threshold >1.5 confirms the mechanism triggers even with the
        # simplified variance approximation (see bdstest docstring).
        x0 = 0.31415926
        n = 800
        x = np.empty(n)
        x[0] = x0
        for i in range(1, n):
            x[i] = 4.0 * x[i - 1] * (1.0 - x[i - 1])
        x = x[100:]
        r = bdstest(x, m=3)
        assert abs(r.stat) > 1.5

    def test_bad_epsilon_rejected(self):
        rng = np.random.default_rng(1)
        with pytest.raises(KuantValueError):
            bdstest(rng.normal(size=200), m=2, epsilon=-0.1)


# ---------- Spectral entropy ------------------------------------------


class TestSpectralEntropy:
    def test_white_noise_near_one(self):
        rng = np.random.default_rng(0)
        r = spectralentropy(rng.normal(size=1024))
        assert r.normalized > 0.85

    def test_sinusoid_near_zero(self):
        # Pure sinusoid: power concentrated at one frequency.
        x = np.sin(2 * np.pi * np.arange(1024) / 32.0)
        r = spectralentropy(x)
        assert r.normalized < 0.20

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            spectralentropy(np.arange(20.0))

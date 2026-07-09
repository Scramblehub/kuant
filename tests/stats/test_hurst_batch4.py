"""Tests for kuant.stats v0.6.0 batch 4: Hurst family variants."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.stats import (
    higuchihurst,
    localwhittle,
    mfdfa,
    wavelethurst,
)


# ---------- higuchihurst ----------------------------------------------


class TestHiguchiHurst:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = higuchihurst(rng.normal(size=500))
        assert hasattr(r, "hurst")
        assert hasattr(r, "fractal_dim")

    def test_noise_dim_near_two(self):
        # Higuchi: white noise has D ~ 2, hence H = 2 - D ~ 0.
        rng = np.random.default_rng(1)
        r = higuchihurst(rng.normal(size=2000))
        assert r.fractal_dim > 1.7
        assert abs(r.hurst) < 0.3

    def test_monotone_low_dim(self):
        # Smooth monotone ramp: D near 1, H near 1.
        x = np.arange(1000.0)
        r = higuchihurst(x)
        assert r.fractal_dim < 1.3

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            higuchihurst(np.arange(50.0))

    def test_bad_kmax_rejected(self):
        rng = np.random.default_rng(2)
        with pytest.raises(KuantValueError):
            higuchihurst(rng.normal(size=200), k_max=100)

    def test_summary(self):
        rng = np.random.default_rng(3)
        r = higuchihurst(rng.normal(size=500))
        assert "HiguchiHurstResult" in r.summary()


# ---------- wavelethurst ----------------------------------------------


class TestWaveletHurst:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = wavelethurst(rng.normal(size=1024))
        assert hasattr(r, "hurst")

    def test_noise_hurst_near_half(self):
        rng = np.random.default_rng(1)
        r = wavelethurst(rng.normal(size=4096))
        assert 0.35 < r.hurst < 0.65

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            wavelethurst(np.arange(50.0))

    def test_summary(self):
        rng = np.random.default_rng(2)
        r = wavelethurst(rng.normal(size=1024))
        assert "WaveletHurstResult" in r.summary()


# ---------- mfdfa ------------------------------------------------------


class TestMfdfa:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = mfdfa(rng.normal(size=500))
        assert hasattr(r, "h_q")
        assert r.h_q.size == r.q_values.size

    def test_noise_h2_near_half(self):
        rng = np.random.default_rng(1)
        r = mfdfa(rng.normal(size=2000))
        h2 = r.h_q[np.argmin(np.abs(r.q_values - 2))]
        assert 0.35 < h2 < 0.65

    def test_multifractal_width_nonneg(self):
        rng = np.random.default_rng(2)
        r = mfdfa(rng.normal(size=2000))
        assert r.multifractal_width >= 0

    def test_bad_order_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError):
            mfdfa(rng.normal(size=500), order=0)

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            mfdfa(np.arange(100.0))


# ---------- localwhittle ----------------------------------------------


class TestLocalWhittle:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = localwhittle(rng.normal(size=1024))
        assert hasattr(r, "d")
        assert hasattr(r, "hurst")

    def test_noise_d_near_zero(self):
        rng = np.random.default_rng(1)
        r = localwhittle(rng.normal(size=4096))
        assert abs(r.d) < 0.15  # noise has d = 0

    def test_hurst_offset(self):
        rng = np.random.default_rng(2)
        r = localwhittle(rng.normal(size=2000))
        assert abs(r.hurst - r.d - 0.5) < 1e-9

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            localwhittle(np.arange(100.0))

    def test_se_positive(self):
        rng = np.random.default_rng(3)
        r = localwhittle(rng.normal(size=2000))
        assert r.se > 0

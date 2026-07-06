"""Tests for kuant.sindy.chaos.corrdim."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.corrdim import CorrDimResult, corrdim


class TestCorrDim:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = corrdim(rng.normal(size=500), tau=1, m=3, n_r=15)
        assert isinstance(r, CorrDimResult)
        assert r.log_r.size == r.log_C.size

    def test_gaussian_dim_grows_with_m(self):
        # Stochastic Gaussian noise: D_2 rises with m (never saturates).
        rng = np.random.default_rng(1)
        x = rng.normal(size=600)
        d3 = corrdim(x, tau=1, m=3, n_r=20).correlation_dim
        d6 = corrdim(x, tau=1, m=6, n_r=20).correlation_dim
        assert d6 > d3

    def test_sinusoid_dim_below_two(self):
        # A pure sinusoid lies on a 1D closed curve; embedded, D_2 -> 1.
        n = 600
        x = np.sin(2 * np.pi * np.arange(n) / 25.0)
        r = corrdim(x, tau=1, m=3, n_r=20)
        # Allow some finite-sample slack.
        assert r.correlation_dim < 2.0

    def test_2d_x_rejected(self):
        with pytest.raises(KuantValueError):
            corrdim(np.zeros((10, 3)), tau=1, m=3)

    def test_too_few_rejected(self):
        with pytest.raises(KuantValueError):
            corrdim(np.arange(100.0), tau=1, m=3)

    def test_bad_range_rejected(self):
        rng = np.random.default_rng(2)
        with pytest.raises(KuantValueError, match=r"r_frac_range"):
            corrdim(rng.normal(size=400), tau=1, m=3, r_frac_range=(0.5, 0.1))

    def test_summary(self):
        rng = np.random.default_rng(3)
        r = corrdim(rng.normal(size=400), tau=1, m=3, n_r=10)
        s = r.summary()
        assert "CorrDimResult" in s
        assert "D_2" in s

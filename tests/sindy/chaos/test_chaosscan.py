"""Tests for kuant.sindy.chaos.chaosscan (composer + classifier)."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.chaosscan import ChaosScanResult, chaosscan


class TestChaosScan:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = chaosscan(rng.normal(size=400), max_lag=10, max_dim=4)
        assert isinstance(r, ChaosScanResult)
        assert r.regime in {"chaotic", "periodic", "stochastic", "unknown"}

    def test_stochastic_or_unknown_for_noise(self):
        rng = np.random.default_rng(1)
        r = chaosscan(rng.normal(size=500), max_lag=10, max_dim=5)
        # Should not misclassify Gaussian noise as chaotic or periodic.
        assert r.regime in {"stochastic", "unknown"}

    def test_periodic_signal_low_dim(self):
        # A pure sinusoid: D_2 should be small and structure (DET / LAM)
        # high. Fix (tau, m) since the auto-picker can over-embed a 1D
        # attractor and fragment diagonals.
        n = 500
        x = np.sin(2 * np.pi * np.arange(n) / 20.0)
        r = chaosscan(x, tau=5, m=2, max_lag=10, max_dim=5)
        # Correlation dim under 2 for a 1D closed curve.
        assert r.corrdim.correlation_dim < 2.5
        # Structure (either diagonals or verticals) should dominate.
        assert max(r.rqa.determinism, r.rqa.laminarity) > 0.7

    def test_explicit_tau_and_m(self):
        rng = np.random.default_rng(2)
        r = chaosscan(rng.normal(size=500), tau=2, m=3, max_dim=5)
        assert r.embed_tau == 2
        assert r.embed_dim == 3

    def test_too_short_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError):
            chaosscan(rng.normal(size=100))

    def test_summary(self):
        rng = np.random.default_rng(4)
        r = chaosscan(rng.normal(size=500), max_lag=10, max_dim=4)
        s = r.summary()
        assert "ChaosScanResult" in s
        assert "regime" in s

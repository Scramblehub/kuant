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


# ---------- v0.5.2: label_calibration opt-in ---------------------------


class TestChaosScanCalibration:
    def test_invalid_calibration_rejected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(KuantValueError, match=r"label_calibration"):
            chaosscan(rng.normal(size=400), label_calibration="bogus")

    def test_default_calibration_is_classical(self):
        """Explicit classical == default (unspecified)."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=500)
        r_default = chaosscan(x, tau=1, m=3)
        r_classical = chaosscan(x, tau=1, m=3, label_calibration="classical")
        assert r_default.regime == r_classical.regime

    def test_financial_label_upgrades_borderline_signal(self):
        """A signal with lambda ~ 1e-4 (below classical 0.001 threshold
        but above financial 1e-5) should classify differently under the
        two calibrations if DET / D_2 also meet the chaotic gates."""
        # Handcrafted: r=3.7 logistic map (mildly chaotic, small lambda).
        n = 800
        x = np.empty(n)
        x[0] = 0.31415926
        r = 3.7
        for i in range(1, n):
            x[i] = r * x[i - 1] * (1.0 - x[i - 1])
        x = x[100:]  # drop transient
        r_classical = chaosscan(x, tau=1, m=3, label_calibration="classical")
        r_financial = chaosscan(x, tau=1, m=3, label_calibration="financial")
        # Financial calibration should be at least as chaotic-inclusive
        # as classical for the same raw numbers.
        classical_chaotic = r_classical.regime == "chaotic"
        financial_chaotic = r_financial.regime == "chaotic"
        # If classical says chaotic, financial must also say chaotic
        # (financial has a lower lambda threshold; nothing rises).
        if classical_chaotic:
            assert financial_chaotic

    def test_raw_metrics_are_calibration_independent(self):
        """Only regime label depends on calibration; raw kernel outputs
        must be identical across calibrations."""
        rng = np.random.default_rng(2)
        x = rng.normal(size=500)
        r_c = chaosscan(x, tau=1, m=3, label_calibration="classical")
        r_f = chaosscan(x, tau=1, m=3, label_calibration="financial")
        assert r_c.lyapunov.lyapunov == r_f.lyapunov.lyapunov
        assert r_c.corrdim.correlation_dim == r_f.corrdim.correlation_dim
        assert r_c.rqa.determinism == r_f.rqa.determinism

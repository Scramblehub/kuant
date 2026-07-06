"""Tests for kuant.sindy.chaos.rqa."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.rqa import RQAResult, rqa


class TestRQA:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = rqa(rng.normal(size=400), tau=1, m=3)
        assert isinstance(r, RQAResult)

    def test_recurrence_rate_in_unit(self):
        rng = np.random.default_rng(1)
        r = rqa(rng.normal(size=400), tau=1, m=3)
        assert 0.0 <= r.recurrence_rate <= 1.0

    def test_determinism_higher_for_periodic(self):
        rng = np.random.default_rng(2)
        n = 400
        periodic = np.sin(2 * np.pi * np.arange(n) / 25.0)
        noise = rng.normal(size=n)
        r_per = rqa(periodic, tau=1, m=3)
        r_noise = rqa(noise, tau=1, m=3)
        # Deterministic sinusoid should show higher DET than i.i.d. noise.
        assert r_per.determinism > r_noise.determinism

    def test_2d_rejected(self):
        with pytest.raises(KuantValueError):
            rqa(np.zeros((10, 3)), tau=1, m=3)

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            rqa(np.arange(50.0), tau=1, m=3)

    def test_explicit_epsilon(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=400)
        r_auto = rqa(x, tau=1, m=3)
        # Passing an explicit epsilon roughly equal to the auto pick
        # should not crash and should return a similar RR.
        r_explicit = rqa(x, tau=1, m=3, epsilon=r_auto.epsilon)
        assert abs(r_auto.recurrence_rate - r_explicit.recurrence_rate) < 1e-9

    def test_summary(self):
        rng = np.random.default_rng(4)
        r = rqa(rng.normal(size=300), tau=1, m=3)
        s = r.summary()
        assert "RQAResult" in s
        assert "determinism" in s

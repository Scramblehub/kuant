"""Tests for kuant.sindy.chaos.lyapunov."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.lyapunov import LyapunovResult, lyapunov


def _logistic(n, r=4.0, x0=0.31415926):
    """Fully chaotic r=4 logistic map.

    Avoid x0 = 0.5 (fixed-point trap: 0.5 -> 1 -> 0 -> 0 -> ...) and
    any dyadic-rational start that collapses to the periodic orbit
    under FP arithmetic.
    """
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    # Drop transient.
    return x[100:]


class TestLyapunov:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = lyapunov(rng.normal(size=400), tau=1, m=3)
        assert isinstance(r, LyapunovResult)

    def test_logistic_map_positive_lyapunov(self):
        # The r=4 fully-chaotic logistic map has lambda = ln(2) ~ 0.693
        # per iteration. Rosenstein on ~700 points should recover a
        # positive slope even with finite-sample noise.
        x = _logistic(900, r=4.0)
        r = lyapunov(x, tau=1, m=3, max_t=15, fit_start=1, fit_end=6)
        assert r.lyapunov > 0

    def test_constant_series_zero_or_negative(self):
        # A constant (with tiny numerical noise) should NOT show
        # positive Lyapunov.
        rng = np.random.default_rng(1)
        x = np.ones(500) + 1e-9 * rng.normal(size=500)
        # Not raising is enough; check slope is not strongly positive.
        r = lyapunov(x, tau=1, m=3, max_t=15, fit_start=1, fit_end=8)
        assert r.lyapunov < 0.2  # not obviously chaotic

    def test_2d_x_rejected(self):
        with pytest.raises(KuantValueError):
            lyapunov(np.zeros((10, 3)), tau=1, m=3)

    def test_too_few_rejected(self):
        with pytest.raises(KuantValueError, match=r"finite"):
            lyapunov(np.arange(50.0), tau=1, m=3)

    def test_bad_m_rejected(self):
        rng = np.random.default_rng(2)
        with pytest.raises(KuantValueError):
            lyapunov(rng.normal(size=400), tau=1, m=1)

    def test_summary(self):
        rng = np.random.default_rng(3)
        r = lyapunov(rng.normal(size=400), tau=1, m=3)
        s = r.summary()
        assert "LyapunovResult" in s
        assert "lambda" in s

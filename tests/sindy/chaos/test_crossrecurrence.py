"""Tests for kuant.sindy.chaos.crossrecurrence (2 kernels)."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.crossrecurrence import (
    CrossRecurrenceResult,
    JointRecurrenceResult,
    crossrecurrence,
    jointrecurrence,
)


class TestCrossRecurrence:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        y = rng.normal(size=400)
        r = crossrecurrence(x, y, tau=1, m=3)
        assert isinstance(r, CrossRecurrenceResult)

    def test_recurrence_rate_hits_target(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        r = crossrecurrence(x, y, tau=1, m=3, recurrence_rate_target=0.10)
        # Auto-epsilon should place RR very close to target.
        assert abs(r.recurrence_rate - 0.10) < 0.01

    def test_identical_series_high_rr(self):
        # x == y should produce very high recurrence.
        rng = np.random.default_rng(2)
        x = rng.normal(size=400)
        r = crossrecurrence(x, x, tau=1, m=3)
        assert r.recurrence_rate > 0.05

    def test_unequal_length_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError, match=r"equal length"):
            crossrecurrence(rng.normal(size=500), rng.normal(size=400))

    def test_too_short_rejected(self):
        rng = np.random.default_rng(4)
        with pytest.raises(KuantValueError):
            crossrecurrence(rng.normal(size=50), rng.normal(size=50))


class TestJointRecurrence:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        y = rng.normal(size=400)
        r = jointrecurrence(x, y, tau=1, m=3)
        assert isinstance(r, JointRecurrenceResult)

    def test_independent_series_product_rule(self):
        # For independent x, y at RR_target = 0.10 each, joint RR should
        # be approximately 0.10 * 0.10 = 0.01.
        rng = np.random.default_rng(1)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        r = jointrecurrence(x, y, tau=1, m=3, recurrence_rate_target=0.10)
        # Allow a wide tolerance because it's a small-sample bias-y estimate.
        assert 0.001 < r.recurrence_rate < 0.04

    def test_identical_series_matches_own_rqa(self):
        # x == y: joint recurrence equals x's own recurrence.
        rng = np.random.default_rng(2)
        x = rng.normal(size=500)
        r = jointrecurrence(x, x, tau=1, m=3, recurrence_rate_target=0.10)
        # Joint RR should be close to the marginal target since the two
        # RPs are identical.
        assert abs(r.recurrence_rate - 0.10) < 0.03

    def test_unequal_length_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError, match=r"equal length"):
            jointrecurrence(rng.normal(size=400), rng.normal(size=300))

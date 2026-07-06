"""Tests for kuant.sindy.chaos.ccm."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.ccm import CCMResult, ccm


class TestCCM:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        y = rng.normal(size=400)
        r = ccm(x, y, tau=1, m=3, n_seeds=2)
        assert isinstance(r, CCMResult)
        assert r.rho_xy.shape == r.rho_yx.shape == r.lib_sizes.shape

    def test_unequal_lengths_rejected(self):
        rng = np.random.default_rng(1)
        with pytest.raises(KuantValueError, match=r"same length"):
            ccm(rng.normal(size=400), rng.normal(size=300), tau=1, m=3)

    def test_too_short_rejected(self):
        rng = np.random.default_rng(2)
        with pytest.raises(KuantValueError):
            ccm(rng.normal(size=100), rng.normal(size=100), tau=1, m=3)

    def test_bad_lib_sizes_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError, match=r"library sizes"):
            ccm(
                rng.normal(size=300),
                rng.normal(size=300),
                tau=1,
                m=3,
                lib_sizes=[1],
            )

    def test_independent_series_no_causality(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=400)
        y = rng.normal(size=400)
        r = ccm(x, y, tau=1, m=3, n_seeds=2)
        # Two independent Gaussians should not both show convergent
        # coupling in both directions.
        assert not (r.convergence_xy and r.convergence_yx)

    def test_summary(self):
        rng = np.random.default_rng(5)
        r = ccm(rng.normal(size=300), rng.normal(size=300), tau=1, m=3, n_seeds=2)
        s = r.summary()
        assert "CCMResult" in s
        assert "convergent" in s

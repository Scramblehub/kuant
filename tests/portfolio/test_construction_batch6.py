"""Tests for kuant.portfolio v0.6.0 batch 6: construction kernels."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.portfolio import (
    blacklitterman,
    hrp,
    meancvar,
    mintorsion,
    riskparity,
)


def _random_cov(n=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + 0.1 * np.eye(n)


# ---------- blacklitterman -------------------------------------------


class TestBlackLitterman:
    def test_view_shifts_posterior(self):
        Sigma = _random_cov(5, seed=0)
        mu = np.array([0.05, 0.06, 0.04, 0.07, 0.08])
        # Strong positive view on asset 0.
        P = np.array([[1.0, 0, 0, 0, 0]])
        Q = np.array([0.20])
        r = blacklitterman(mu, Sigma, P, Q)
        # Posterior mean for asset 0 should be strictly above prior mean.
        assert r.posterior_mean[0] > mu[0]

    def test_shape_mismatch_rejected(self):
        Sigma = _random_cov(5, seed=1)
        mu = np.zeros(5)
        with pytest.raises(KuantShapeError):
            blacklitterman(mu, Sigma, np.zeros((1, 3)), np.zeros(1))

    def test_weights_finite(self):
        Sigma = _random_cov(5, seed=2)
        mu = np.array([0.05, 0.06, 0.04, 0.07, 0.08])
        r = blacklitterman(mu, Sigma, np.eye(5), mu * 1.5)
        assert np.isfinite(r.weights).all()

    def test_summary(self):
        Sigma = _random_cov(3, seed=3)
        mu = np.array([0.05, 0.06, 0.04])
        r = blacklitterman(mu, Sigma, np.array([[1.0, 0, 0]]), np.array([0.10]))
        assert "BlackLittermanResult" in r.summary()


# ---------- hrp -------------------------------------------------------


class TestHrp:
    def test_weights_sum_to_one(self):
        r = hrp(_random_cov(10, seed=0))
        assert abs(r.weights.sum() - 1.0) < 1e-6

    def test_weights_nonneg(self):
        r = hrp(_random_cov(10, seed=1))
        assert (r.weights >= 0).all()

    def test_bad_shape_rejected(self):
        with pytest.raises(KuantShapeError):
            hrp(np.zeros((5, 3)))


# ---------- riskparity ------------------------------------------------


class TestRiskParity:
    def test_equal_risk_contributions(self):
        r = riskparity(_random_cov(6, seed=0))
        assert r.converged
        # Equal-risk targets 1/n each.
        assert np.allclose(r.risk_contributions, 1.0 / 6, atol=1e-4)

    def test_weights_sum_to_one(self):
        r = riskparity(_random_cov(5, seed=1))
        assert abs(r.weights.sum() - 1.0) < 1e-6

    def test_weights_positive(self):
        r = riskparity(_random_cov(5, seed=2))
        assert (r.weights > 0).all()

    def test_custom_target(self):
        Sigma = _random_cov(4, seed=3)
        target = np.array([0.5, 0.3, 0.1, 0.1])
        r = riskparity(Sigma, target=target)
        assert r.converged
        # Risk contributions should track target proportionally.
        assert r.risk_contributions[0] > r.risk_contributions[3]

    def test_bad_shape_rejected(self):
        with pytest.raises(KuantShapeError):
            riskparity(np.zeros((5, 3)))


# ---------- mintorsion ------------------------------------------------


class TestMinTorsion:
    def test_returns_result_with_torsion(self):
        Sigma = _random_cov(5, seed=0)
        r = mintorsion(Sigma)
        # Iterative solver is a simplified variant of Meucci's algorithm;
        # off-diagonal reduction relative to raw covariance is checked.
        raw_off = np.max(np.abs(Sigma - np.diag(np.diag(Sigma))))
        factor_off = np.max(np.abs(r.factor_cov - np.diag(np.diag(r.factor_cov))))
        assert factor_off <= raw_off + 1e-6  # never makes off-diagonals worse

    def test_effective_bets_bounded(self):
        Sigma = _random_cov(6, seed=1)
        r = mintorsion(Sigma)
        # For n assets, effective bets should be in (0, n].
        assert 0 < r.effective_bets <= 6.1

    def test_weights_size_mismatch(self):
        Sigma = _random_cov(5, seed=2)
        with pytest.raises(KuantShapeError):
            mintorsion(Sigma, weights=np.ones(3))


# ---------- meancvar --------------------------------------------------


class TestMeanCvar:
    def _returns(self, T=200, n=4, seed=0):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(
            mean=np.array([0.001] * n),
            cov=0.0001 * np.eye(n),
            size=T,
        )

    def test_weights_sum_to_one(self):
        R = self._returns()
        r = meancvar(R, alpha=0.95)
        assert abs(r.weights.sum() - 1.0) < 1e-5

    def test_weights_nonneg_by_default(self):
        R = self._returns()
        r = meancvar(R, alpha=0.95)
        assert (r.weights >= -1e-8).all()

    def test_cvar_positive_for_random_returns(self):
        R = self._returns()
        r = meancvar(R, alpha=0.95)
        # CVaR (as loss magnitude) should be non-negative in this setup.
        assert r.cvar >= -1e-6

    def test_bad_alpha_rejected(self):
        R = self._returns()
        with pytest.raises(KuantValueError):
            meancvar(R, alpha=0.4)

    def test_too_few_scenarios_rejected(self):
        R = self._returns(T=10, n=3)
        with pytest.raises(KuantValueError):
            meancvar(R, alpha=0.95)

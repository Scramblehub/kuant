"""Tests for kuant.qm.ghmm.baumwelch — EM training for Gaussian HMMs."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.qm.ghmm.baumwelch import GHMMBaumWelchResult, baumwelch


def _sample_ghmm(pi, A, mu, sigma, T, rng):
    """Draw a length-T observation sequence from a Gaussian HMM."""
    pi = np.asarray(pi)
    A = np.asarray(A)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    N = pi.size
    states = np.empty(T, dtype=np.int64)
    obs = np.empty(T, dtype=np.float64)
    states[0] = int(rng.choice(N, p=pi))
    obs[0] = rng.normal(mu[states[0]], sigma[states[0]])
    for t in range(1, T):
        states[t] = int(rng.choice(N, p=A[states[t - 1]]))
        obs[t] = rng.normal(mu[states[t]], sigma[states[t]])
    return obs, states


# ---------- return-object contract ----------------------------------------


def test_returns_ghmm_baumwelch_result():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=200)
    r = baumwelch(obs, n_states=2, seed=0, max_iter=10)
    assert isinstance(r, GHMMBaumWelchResult)
    assert r.pi.shape == (2,)
    assert r.A.shape == (2, 2)
    assert r.mu.shape == (2,)
    assert r.sigma.shape == (2,)


def test_output_stochastic_and_positive_sigma():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=300)
    r = baumwelch(obs, n_states=3, seed=0, max_iter=20)
    assert abs(r.pi.sum() - 1.0) < 1e-9
    assert np.allclose(r.A.sum(axis=1), 1.0)
    assert (r.sigma > 0).all()


# ---------- Baum's inequality ---------------------------------------------


def test_log_likelihood_monotone_nondecreasing():
    rng = np.random.default_rng(0)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu = np.array([0.0, 3.0])
    sigma = np.array([1.0, 1.5])
    obs, _ = _sample_ghmm(pi, A, mu, sigma, T=300, rng=rng)
    r = baumwelch(obs, n_states=2, seed=1, max_iter=25)
    diffs = np.diff(r.log_likelihood_history)
    assert (diffs >= -1e-9).all(), (
        f"logL dropped at iter {np.where(diffs < 0)[0]}, " f"min diff {diffs.min():.2e}"
    )


# ---------- ground-truth recovery -----------------------------------------


def test_recovers_well_separated_regimes():
    """Two well-separated Gaussian regimes → EM recovers μ and σ within
    a coarse tolerance (up to state permutation)."""
    rng = np.random.default_rng(42)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu = np.array([0.0, 5.0])  # far apart — well-identified
    sigma = np.array([1.0, 1.0])
    obs, _ = _sample_ghmm(pi, A, mu, sigma, T=600, rng=rng)

    # Two seeds; keep best.
    best = None
    for seed in range(2):
        r = baumwelch(obs, n_states=2, seed=seed, max_iter=50, tol=1e-4)
        if best is None or r.log_likelihood > best.log_likelihood:
            best = r

    # Sort estimated μ ascending to align with truth.
    order = np.argsort(best.mu)
    mu_est = best.mu[order]
    sigma_est = best.sigma[order]
    assert np.max(np.abs(mu_est - mu)) < 0.5
    assert np.max(np.abs(sigma_est - sigma)) < 0.3


def test_warm_start_at_truth_converges():
    """Warm-start at truth → converges quickly and remains near truth."""
    rng = np.random.default_rng(0)
    pi = np.array([0.6, 0.4])
    A = np.array([[0.9, 0.1], [0.15, 0.85]])
    mu = np.array([0.0, 2.0])
    sigma = np.array([1.0, 1.0])
    obs, _ = _sample_ghmm(pi, A, mu, sigma, T=400, rng=rng)

    r = baumwelch(
        obs,
        pi_init=pi,
        A_init=A,
        mu_init=mu,
        sigma_init=sigma,
        max_iter=30,
        tol=1e-3,
    )
    assert r.converged
    # Sorted μ should stay close to truth.
    mu_sorted = np.sort(r.mu)
    assert np.max(np.abs(mu_sorted - np.sort(mu))) < 0.3


# ---------- σ floor + state collapse --------------------------------------


def test_sigma_floor_prevents_collapse():
    """When one state's σ would want to shrink to zero, the floor holds."""
    rng = np.random.default_rng(0)
    # Series with a repeated near-constant span that EM would try to
    # collapse a state onto.
    obs = np.concatenate(
        [
            rng.normal(0.0, 1.0, 100),
            np.full(15, 5.0) + rng.normal(0, 1e-8, 15),  # tight cluster
            rng.normal(0.0, 1.0, 100),
        ]
    )
    r = baumwelch(obs, n_states=2, seed=0, max_iter=40, min_sigma=1e-3)
    assert (r.sigma >= 1e-3 - 1e-12).all(), r.sigma


def test_reject_negative_initial_sigma():
    obs = np.random.default_rng(0).normal(size=50)
    with pytest.raises(KuantValueError) as exc:
        baumwelch(
            obs,
            pi_init=[0.5, 0.5],
            A_init=[[0.9, 0.1], [0.1, 0.9]],
            mu_init=[0.0, 1.0],
            sigma_init=[1.0, -0.5],
            max_iter=10,
        )
    assert "sigma" in str(exc.value)


# ---------- convergence flag ----------------------------------------------


def test_hits_max_iter_when_tol_too_tight():
    obs = np.random.default_rng(0).normal(size=200)
    r = baumwelch(obs, n_states=2, seed=0, max_iter=3, tol=1e-20)
    assert r.n_iter == 3
    assert not r.converged


# ---------- error contract ------------------------------------------------


def test_reject_2d_obs():
    with pytest.raises(KuantValueError):
        baumwelch(np.zeros((10, 3)), n_states=2)


def test_reject_singleton_obs():
    with pytest.raises(KuantValueError) as exc:
        baumwelch(np.array([0.5]), n_states=2)
    assert "length >= 2" in str(exc.value)


def test_reject_nan_obs():
    obs = np.array([1.0, 2.0, np.nan, 3.0])
    with pytest.raises(KuantValueError) as exc:
        baumwelch(obs, n_states=2)
    m = str(exc.value)
    assert "non-finite" in m or "NaN" in m


def test_reject_mutex_init():
    obs = np.zeros(10)
    with pytest.raises(KuantValueError) as exc:
        baumwelch(obs)
    m = str(exc.value)
    assert "n_states" in m and "pi_init" in m


def test_reject_zero_max_iter():
    obs = np.zeros(10)
    with pytest.raises(KuantValueError):
        baumwelch(obs, n_states=2, max_iter=0)


def test_summary_string_shape():
    obs = np.random.default_rng(0).normal(size=100)
    r = baumwelch(obs, n_states=2, seed=0, max_iter=10)
    s = r.summary()
    assert "BaumWelch" in s and "Gaussian" in s
    assert "log-likelihood" in s
    assert "σ-floor" in s

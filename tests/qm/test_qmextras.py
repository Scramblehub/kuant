'''Tests for posteriorentropy, nocloningscan, decoherencescan, and ghmm.'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.qm import (
    decoherencescan, ghmm, nocloningscan, posteriorentropy,
)


# ---------------------------------------------------------------------------
# posteriorentropy
# ---------------------------------------------------------------------------


def test_posteriorentropy_confident_row_low_entropy():
    gamma = np.array([[0.99, 0.01], [0.5, 0.5], [0.9, 0.1]])
    result = posteriorentropy(gamma)
    assert result.entropy[0] < result.entropy[1]  # first confident, second uniform
    assert result.entropy[2] < result.entropy[1]  # third also confident


def test_posteriorentropy_max_entropy_matches_uniform():
    N = 4
    gamma = np.full((3, N), 1.0 / N)
    result = posteriorentropy(gamma)
    np.testing.assert_allclose(result.entropy, np.log(N), atol=1e-12)
    assert result.max_entropy == pytest.approx(np.log(N))


def test_posteriorentropy_per_regime():
    gamma = np.array([[0.99, 0.01], [0.99, 0.01], [0.5, 0.5], [0.5, 0.5]])
    regime = np.array(['low', 'low', 'high', 'high'])
    result = posteriorentropy(gamma, regime=regime)
    assert result.per_regime is not None
    assert result.per_regime['low']['mean'] < result.per_regime['high']['mean']


def test_posteriorentropy_dimension_mismatch_raises():
    gamma = np.random.default_rng(0).dirichlet(np.ones(2), size=5)
    with pytest.raises(ValueError, match='regime length'):
        posteriorentropy(gamma, regime=np.array(['a', 'b']))


def test_posteriorentropy_summary_readable():
    gamma = np.array([[0.9, 0.1], [0.5, 0.5]])
    text = posteriorentropy(gamma).summary()
    assert 'entropy' in text.lower()


# ---------------------------------------------------------------------------
# nocloningscan
# ---------------------------------------------------------------------------


def test_nocloningscan_high_pair_corr_when_identical():
    '''If fit_predict_fn returns identical predictions across seeds,
    the pair correlation should be near 1.'''
    def fit_predict(seed):
        return np.arange(50, dtype=float), {'m': 1.0}
    result = nocloningscan(fit_predict, n_seeds=3)
    assert result.prediction_pair_corr_mean == pytest.approx(1.0)


def test_nocloningscan_verdict_matches_stats():
    def fit_predict(seed):
        rng = np.random.default_rng(seed)
        return rng.normal(size=100), {'r2': 0.5 + rng.normal(scale=0.001)}
    result = nocloningscan(fit_predict, n_seeds=5)
    # Different paths (pair corr low), tight metric (CV small).
    assert result.prediction_pair_corr_mean < 0.5
    assert result.metric_stats['r2']['cv'] < 0.01
    assert 'DIFFERENT PATHS' in result.summary()


def test_nocloningscan_min_seeds():
    with pytest.raises(ValueError, match='must be >= 2'):
        nocloningscan(lambda s: (np.array([0.0]), {}), n_seeds=1)


# ---------------------------------------------------------------------------
# decoherencescan
# ---------------------------------------------------------------------------


def test_decoherencescan_basic_shape():
    rng = np.random.default_rng(0)
    T = 400
    X = rng.normal(size=(T, 2))
    y = X @ [0.3, -0.2] + rng.normal(scale=0.5, size=T)

    def fit_fn(Xt, yt):
        return np.linalg.lstsq(Xt, yt, rcond=None)[0]

    def predict_fn(model, X_bar):
        return X_bar @ model

    result = decoherencescan(
        fit_fn, predict_fn, X, y,
        train_window=100, predict_window=50,
    )
    assert len(result.bucket_corr) == len(result.bucket_bounds)
    assert result.peak_bucket_idx >= 0
    assert result.peak_bucket_idx < len(result.bucket_bounds)


def test_decoherencescan_summary_readable():
    rng = np.random.default_rng(0)
    T = 300
    X = rng.normal(size=(T, 2))
    y = X @ [0.5, 0.0] + rng.normal(scale=0.3, size=T)

    def fit_fn(Xt, yt):
        return np.linalg.lstsq(Xt, yt, rcond=None)[0]

    def predict_fn(m, X_bar):
        return X_bar @ m

    result = decoherencescan(fit_fn, predict_fn, X, y, train_window=100, predict_window=40)
    text = result.summary()
    assert 'decoherence' in text.lower()


# ---------------------------------------------------------------------------
# ghmm
# ---------------------------------------------------------------------------


@pytest.fixture
def ghmm_2state():
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([0.0, 3.0])
    sigma = np.array([1.0, 1.0])
    return pi, A, mu, sigma


def test_ghmm_forward_backward_likelihood_match(ghmm_2state):
    pi, A, mu, sigma = ghmm_2state
    obs = np.array([0.1, 0.2, 3.1, 2.9, 0.0])
    _, log_lik_fwd = ghmm.forward(obs, pi, A, mu, sigma)
    log_beta = ghmm.backward(obs, pi, A, mu, sigma)
    from scipy.special import logsumexp
    # log P(O) via backward: logsumexp_i(log_pi[i] + log_B[0, i] + log_beta[0, i])
    log_pi = np.log(pi)
    z0 = (obs[0] - mu) / sigma
    log_B0 = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * z0 * z0
    log_lik_bwd = float(logsumexp(log_pi + log_B0 + log_beta[0]))
    assert abs(log_lik_fwd - log_lik_bwd) < 1e-10


def test_ghmm_viterbi_recovers_regime(ghmm_2state):
    '''With well-separated means, Viterbi should track the true regime.'''
    pi, A, mu, sigma = ghmm_2state
    # Two clear regimes: first 5 near 0, last 5 near 3
    obs = np.array([0.1, -0.2, 0.1, 0.3, -0.1, 3.0, 2.9, 3.1, 2.8, 3.2])
    states, _ = ghmm.viterbi(obs, pi, A, mu, sigma)
    assert np.all(states[:5] == 0)
    assert np.all(states[5:] == 1)


def test_ghmm_posterior_gamma_rows_sum_to_one(ghmm_2state):
    pi, A, mu, sigma = ghmm_2state
    obs = np.array([0.1, 3.0, 0.2, 2.9])
    gamma, xi, _ = ghmm.posterior(obs, pi, A, mu, sigma)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-12)


def test_ghmm_negative_sigma_raises(ghmm_2state):
    pi, A, mu, sigma = ghmm_2state
    obs = np.array([0.1, 0.2])
    bad_sigma = np.array([1.0, -0.5])
    with pytest.raises(ValueError, match='sigma must be positive'):
        ghmm.forward(obs, pi, A, mu, bad_sigma)


def test_ghmm_shape_mismatch_raises(ghmm_2state):
    pi, A, mu, sigma = ghmm_2state
    obs = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match='mu must be'):
        ghmm.forward(obs, pi, A, np.array([0.0]), sigma)

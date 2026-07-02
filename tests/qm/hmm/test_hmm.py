'''Test suite for kuant.qm.hmm — forward, backward, viterbi, posterior.'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.qm.hmm import backward, forward, posterior, viterbi


@pytest.fixture
def hmm_2state():
    '''Textbook 2-state HMM (rainy/sunny → walk/shop/clean).'''
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    return pi, A, B


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_forward_shape_and_likelihood(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0, 1])
    log_alpha, log_lik = forward(obs, pi, A, B)
    assert log_alpha.shape == (5, 2)
    # Log-likelihood should be < 0 (probabilities <= 1)
    assert log_lik <= 0.0


def test_forward_single_observation(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0])
    log_alpha, log_lik = forward(obs, pi, A, B)
    # log_alpha[0, i] = log(pi[i]) + log(B[i, 0])
    expected = np.log(pi) + np.log(B[:, 0])
    np.testing.assert_allclose(log_alpha[0], expected, atol=1e-12)


def test_forward_dimension_mismatch_raises(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1])
    with pytest.raises(ValueError, match=r'A must be'):
        forward(obs, pi, A[:1], B)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


def test_backward_shape(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0, 1])
    log_beta = backward(obs, pi, A, B)
    assert log_beta.shape == (5, 2)


def test_backward_last_row_is_zero(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2])
    log_beta = backward(obs, pi, A, B)
    # β[T-1, i] = 1 for all i → log(1) = 0
    np.testing.assert_array_equal(log_beta[-1], [0.0, 0.0])


def test_forward_backward_likelihood_matches(hmm_2state):
    '''log P(O) computed from forward matches log P(O) from backward:
       log P(O) = logsumexp_i(log_pi[i] + log_B[i, o_0] + log_beta[0, i]).'''
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0])
    _, log_lik_fwd = forward(obs, pi, A, B)
    log_beta = backward(obs, pi, A, B)
    log_pi = np.log(pi)
    log_B = np.log(B)
    from scipy.special import logsumexp
    log_lik_bwd = logsumexp(log_pi + log_B[:, obs[0]] + log_beta[0])
    assert abs(log_lik_fwd - log_lik_bwd) < 1e-10


# ---------------------------------------------------------------------------
# viterbi
# ---------------------------------------------------------------------------


def test_viterbi_returns_valid_states(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0, 1, 2])
    states, log_prob = viterbi(obs, pi, A, B)
    assert states.shape == (6,)
    assert np.all(states >= 0)
    assert np.all(states < 2)  # 2 hidden states
    assert log_prob <= 0.0


def test_viterbi_locks_in_when_initial_prior_favors_state():
    '''With strong initial prior on state 0 and near-identity transitions,
    Viterbi should stay in state 0 even after obs favors state 1 briefly —
    because switching costs ~23 log-units.'''
    pi = np.array([0.99, 0.01])                          # strong prior state 0
    A = np.array([[1.0 - 1e-10, 1e-10], [1e-10, 1.0 - 1e-10]])
    B = np.array([[0.7, 0.3], [0.3, 0.7]])               # milder emission
    obs = np.array([0, 0, 0, 1, 1])                       # ends with obs 1
    states, _ = viterbi(obs, pi, A, B)
    # Prior + transition cost keeps us in state 0 despite two contrary obs.
    assert np.all(states == 0)


def test_viterbi_switches_when_evidence_overwhelms_transition_cost():
    '''If emission strongly favors switching and transition cost is small,
    Viterbi should switch states mid-sequence.'''
    pi = np.array([0.5, 0.5])
    A = np.array([[0.6, 0.4], [0.4, 0.6]])                # moderate transitions
    B = np.array([[0.99, 0.01], [0.01, 0.99]])            # strong emission
    obs = np.array([0, 0, 0, 1, 1, 1])
    states, _ = viterbi(obs, pi, A, B)
    # Should be some mix — first half favors state 0, second half favors state 1.
    assert np.any(states == 0) and np.any(states == 1)


# ---------------------------------------------------------------------------
# posterior
# ---------------------------------------------------------------------------


def test_gamma_rows_sum_to_one(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0, 1])
    gamma, xi, log_lik = posterior(obs, pi, A, B)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-12)


def test_xi_slices_sum_to_one(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0, 1])
    _, xi, _ = posterior(obs, pi, A, B)
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-12)


def test_gamma_marginalizes_xi(hmm_2state):
    '''For t < T-1: gamma[t, i] should equal sum_j xi[t, i, j].'''
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0])
    gamma, xi, _ = posterior(obs, pi, A, B)
    marginal = xi.sum(axis=2)  # (T-1, N)
    np.testing.assert_allclose(gamma[:-1], marginal, atol=1e-10)


def test_posterior_log_likelihood_matches_forward(hmm_2state):
    pi, A, B = hmm_2state
    obs = np.array([0, 1, 2, 0])
    _, log_lik_fwd = forward(obs, pi, A, B)
    _, _, log_lik_post = posterior(obs, pi, A, B)
    assert abs(log_lik_fwd - log_lik_post) < 1e-10

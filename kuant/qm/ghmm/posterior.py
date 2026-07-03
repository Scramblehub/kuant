'''Gaussian HMM state posteriors (γ) and joint posteriors (ξ).'''
from __future__ import annotations

import numpy as np

from .backward import backward
from .common import _prepare_ghmm_inputs
from .forward import forward


def posterior(obs, pi, A, mu, sigma):
    '''State posterior γ, joint posterior ξ, and log-likelihood.

    Returns:
        gamma (T, N)     — rows sum to 1
        xi    (T-1, N, N) — per-t slices sum to 1
        log_likelihood
    '''
    xp, obs_arr, log_pi, log_A, log_B = _prepare_ghmm_inputs(obs, pi, A, mu, sigma)
    T = obs_arr.size

    log_alpha, log_lik = forward(obs, pi, A, mu, sigma)
    log_beta = backward(obs, pi, A, mu, sigma)

    log_gamma = log_alpha + log_beta - log_lik
    gamma = np.exp(log_gamma)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    log_xi = (
        log_alpha[:-1, :, None]
        + log_A[None, :, :]
        + log_B[1:, None, :]
        + log_beta[1:, None, :]
        - log_lik
    )
    xi = np.exp(log_xi)
    xi = xi / xi.sum(axis=(1, 2), keepdims=True)

    return gamma, xi, log_lik

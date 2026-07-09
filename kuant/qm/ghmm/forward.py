"""Gaussian HMM forward algorithm (log-space)."""

from __future__ import annotations

import numpy as np

from .common import _logsumexp_axis, _prepare_ghmm_inputs


def forward(obs, pi, A, mu, sigma):
    """Log-space forward for a Gaussian HMM.

    obs   : (T,) continuous observations
    pi    : (N,) initial state distribution
    A     : (N, N) transition matrix
    mu    : (N,) per-state emission mean
    sigma : (N,) per-state emission std (> 0)

    Returns (log_alpha (T, N), log_likelihood).
    """
    xp, obs_arr, log_pi, log_A, log_B = _prepare_ghmm_inputs(obs, pi, A, mu, sigma)
    T = obs_arr.size
    N = log_pi.size

    log_alpha = np.full((T, N), -np.inf, dtype=np.float64)
    log_alpha[0] = log_pi + log_B[0]

    for t in range(1, T):
        combined = log_alpha[t - 1, :, None] + log_A
        log_alpha[t] = _logsumexp_axis(xp, combined, axis=0) + log_B[t]

    log_lik = float(_logsumexp_axis(xp, log_alpha[-1], axis=0))
    return log_alpha, log_lik

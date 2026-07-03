'''Gaussian HMM backward algorithm (log-space).'''
from __future__ import annotations

import numpy as np

from .common import _logsumexp_axis, _prepare_ghmm_inputs


def backward(obs, pi, A, mu, sigma):
    '''Log-space backward for a Gaussian HMM.

    Returns log_beta (T, N).
    '''
    xp, obs_arr, _log_pi, log_A, log_B = _prepare_ghmm_inputs(obs, pi, A, mu, sigma)
    T = obs_arr.size
    N = log_A.shape[0]

    log_beta = np.full((T, N), -np.inf, dtype=np.float64)
    log_beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        combined = log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :]
        log_beta[t] = _logsumexp_axis(xp, combined, axis=1)

    return log_beta

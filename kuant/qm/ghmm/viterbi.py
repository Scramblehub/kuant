'''Gaussian HMM Viterbi decoding.'''
from __future__ import annotations

import numpy as np

from .common import _prepare_ghmm_inputs


def viterbi(obs, pi, A, mu, sigma):
    '''Most-likely state sequence for a Gaussian HMM.

    Returns (states (T,) int, log_prob float).
    '''
    xp, obs_arr, log_pi, log_A, log_B = _prepare_ghmm_inputs(obs, pi, A, mu, sigma)
    T = obs_arr.size
    N = log_pi.size

    log_delta = np.full((T, N), -np.inf, dtype=np.float64)
    psi = np.zeros((T, N), dtype=np.int64)

    log_delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        combined = log_delta[t - 1, :, None] + log_A
        psi[t] = np.argmax(combined, axis=0)
        log_delta[t] = np.max(combined, axis=0) + log_B[t]

    states = np.zeros(T, dtype=np.int64)
    states[T - 1] = int(np.argmax(log_delta[T - 1]))
    log_prob = float(log_delta[T - 1, states[T - 1]])

    for t in range(T - 2, -1, -1):
        states[t] = int(psi[t + 1, int(states[t + 1])])

    return states, log_prob

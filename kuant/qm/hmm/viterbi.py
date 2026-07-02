'''HMM Viterbi decoding — most likely state sequence.

    δ[t, i] = max over paths of P(o_0..o_t, s_0..s_{t-1}, s_t = i | model)
    ψ[t, i] = argmax that produced δ[t, i]

Recursion:
    δ[0, i]   = π[i] · B[i, o_0]
    δ[t, j]   = max_i (δ[t-1, i] · A[i, j]) · B[j, o_t]
    ψ[t, j]   = argmax_i (δ[t-1, i] · A[i, j])

Traceback:
    s_{T-1} = argmax_i δ[T-1, i]
    s_t     = ψ[t+1, s_{t+1}]

Log-space throughout.

Design: docs/kernels/hmm_viterbi.md.
'''
from __future__ import annotations

import numpy as np

from .forward import _prepare_hmm_inputs


def viterbi(obs, pi, A, B):
    '''Most-likely state sequence via Viterbi decoding.

    Returns
    -------
    states : 1D int array, length T
        Most likely state at each time step.
    log_prob : float
        Log-probability of the returned path.
    '''
    xp, obs_arr, log_pi, log_A, log_B = _prepare_hmm_inputs(obs, pi, A, B)
    T = obs_arr.size
    N = log_pi.size

    log_delta = xp.full((T, N), -xp.inf, dtype=np.float64)
    psi = xp.zeros((T, N), dtype=np.int64)

    log_delta[0] = log_pi + log_B[:, obs_arr[0]]

    for t in range(1, T):
        # For each next state j:  score[j, i] = log_delta[t-1, i] + log_A[i, j]
        # Then log_delta[t, j] = max_i score[j, i] + log_B[j, o_t]
        # And psi[t, j] = argmax_i.
        combined = log_delta[t-1, :, None] + log_A          # (N_prev, N_next)
        best_prev = xp.argmax(combined, axis=0)              # (N_next,)
        best_val = xp.max(combined, axis=0)                  # (N_next,)
        log_delta[t] = best_val + log_B[:, obs_arr[t]]
        psi[t] = best_prev

    # Traceback
    states = xp.zeros(T, dtype=np.int64)
    states[T-1] = int(xp.argmax(log_delta[T-1]))
    log_prob = float(log_delta[T-1, states[T-1]])

    for t in range(T - 2, -1, -1):
        states[t] = int(psi[t + 1, int(states[t + 1])])

    return states, log_prob

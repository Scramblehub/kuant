"""HMM backward algorithm (log-space).

    β[t, i] = P(o_{t+1}, ..., o_{T-1} | state_t = i, model)

Recursion:
    β[T-1, i] = 1
    β[t, i]   = Σ_j A[i, j] · B[j, o_{t+1}] · β[t+1, j]

Design: docs/kernels/hmm_backward.md.
"""

from __future__ import annotations

import numpy as np

from .forward import _logsumexp_axis, _prepare_hmm_inputs


def backward(obs, pi, A, B):
    """Log-space HMM backward algorithm.

    Parameters
    ----------
    obs : 1D int array, length T
    pi : 1D array, length N   (unused mathematically but validated + kept for API symmetry)
    A : (N, N)
    B : (N, M)

    Returns
    -------
    log_beta : (T, N) array
        Backward log-probabilities.
    """
    xp, obs_arr, _log_pi, log_A, log_B = _prepare_hmm_inputs(obs, pi, A, B)
    T = obs_arr.size
    N = log_A.shape[0]

    log_beta = xp.full((T, N), -xp.inf, dtype=np.float64)

    # Boundary condition: β[T-1, i] = 1 → log = 0.
    log_beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j(log_A[i, j] + log_B[j, o_{t+1}] + log_beta[t+1, j])
        combined = log_A + log_B[:, obs_arr[t + 1]][None, :] + log_beta[t + 1][None, :]
        log_beta[t] = _logsumexp_axis(xp, combined, axis=1)

    return log_beta

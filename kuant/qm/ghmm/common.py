'''Shared setup for Gaussian HMM inference (continuous scalar observations).

    obs   — 1D float array, length T
    pi    — 1D array of prior probabilities, shape (N,)
    A     — transition matrix, shape (N, N), rows sum to 1
    mu    — per-state emission means, shape (N,)
    sigma — per-state emission std devs, shape (N,), > 0

Emission likelihood per state:
    B[t, i] = φ((obs[t] - mu[i]) / sigma[i]) / sigma[i]

which is the normal PDF `N(mu[i], sigma[i]²)` evaluated at obs[t].
Computed in log-space directly to avoid ever exponentiating tiny values.
'''
from __future__ import annotations

from typing import Any

import numpy as np

from ..hmm.forward import _logsumexp_axis   # reuse

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _prepare_ghmm_inputs(obs, pi, A, mu, sigma):
    '''Validate + coerce inputs; precompute log_pi, log_A, log_B.

    log_B[t, i] = log N(obs[t]; mu[i], sigma[i]²)
                = -0.5·log(2π) - log(sigma[i]) - 0.5·((obs[t] - mu[i])/sigma[i])²
    '''
    obs_arr = np.asarray(obs, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    A_arr = np.asarray(A, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    if obs_arr.ndim != 1:
        raise ValueError(f'obs must be 1D, got shape {obs_arr.shape}')
    if pi_arr.ndim != 1:
        raise ValueError(f'pi must be 1D, got shape {pi_arr.shape}')
    N = pi_arr.size
    if A_arr.shape != (N, N):
        raise ValueError(f'A must be ({N}, {N}), got {A_arr.shape}')
    if mu_arr.shape != (N,):
        raise ValueError(f'mu must be ({N},), got {mu_arr.shape}')
    if sigma_arr.shape != (N,):
        raise ValueError(f'sigma must be ({N},), got {sigma_arr.shape}')
    if np.any(sigma_arr <= 0):
        raise ValueError('sigma must be positive per state')

    with np.errstate(divide='ignore'):
        log_pi = np.log(pi_arr)
        log_A = np.log(A_arr)

    # log_B: (T, N) — broadcast obs (T,1) against (mu, sigma) per state.
    z = (obs_arr[:, None] - mu_arr[None, :]) / sigma_arr[None, :]
    log_B = -_LOG_SQRT_2PI - np.log(sigma_arr)[None, :] - 0.5 * z * z

    return np, obs_arr, log_pi, log_A, log_B


__all__ = ['_prepare_ghmm_inputs', '_logsumexp_axis']

"""HMM forward algorithm (log-space).

For a discrete-observation HMM with:
  N hidden states, M observation symbols
  pi[N]     — initial state distribution
  A[N, N]   — transition matrix: A[i, j] = P(state j at t+1 | state i at t)
  B[N, M]   — emission matrix:   B[i, k] = P(observe k | state i)

The forward variable is:
    α[t, i] = P(o_0, o_1, ..., o_t, state_t = i | model)

Recursion:
    α[0, i]    = π[i] · B[i, o_0]
    α[t+1, j]  = (Σ_i α[t, i] · A[i, j]) · B[j, o_{t+1}]

Likelihood:
    P(O | model) = Σ_i α[T-1, i]

We work entirely in log-space with `scipy.special.logsumexp` to avoid
underflow on long sequences.

Design: docs/kernels/hmm_forward.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp

from kuant._validation import require_1d, require_expected_shape

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


def _prepare_hmm_inputs(obs, pi, A, B):
    """Validate and coerce (obs, pi, A, B) into a consistent backend.

    Returns (backend, obs, log_pi, log_A, log_B).
    obs is int (observation indices); the others are log-probabilities.
    """
    if (
        isinstance(pi, _CUPY_NDARRAY)
        or isinstance(A, _CUPY_NDARRAY)
        or isinstance(B, _CUPY_NDARRAY)
    ):
        xp = cp
    else:
        xp = np

    obs_arr = xp.asarray(obs)
    pi_arr = xp.asarray(pi, dtype=np.float64)
    A_arr = xp.asarray(A, dtype=np.float64)
    B_arr = xp.asarray(B, dtype=np.float64)

    require_1d(obs_arr, "obs", kernel="hmm.forward")
    if obs_arr.dtype.kind not in "iu":
        # Allow float ints (e.g. from numpy) but require integer values
        obs_arr = obs_arr.astype(np.int64)
    require_1d(pi_arr, "pi", kernel="hmm.forward")
    N = pi_arr.size
    require_expected_shape(A_arr, "A", (N, N), kernel="hmm.forward")
    require_expected_shape(B_arr, "B", (N, "M"), kernel="hmm.forward")

    # Move to log-space. Guard against log(0) with a floor of -inf.
    with np.errstate(divide="ignore") if xp is np else _null_context():
        log_pi = xp.log(pi_arr)
        log_A = xp.log(A_arr)
        log_B = xp.log(B_arr)

    return xp, obs_arr, log_pi, log_A, log_B


class _null_context:
    """No-op context manager for cupy paths (cupy has no errstate)."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _logsumexp_axis(xp, arr, axis):
    """Numerically-stable logsumexp along an axis, for numpy or cupy."""
    if xp is np:
        return logsumexp(arr, axis=axis)
    # cupy: implement inline
    m = xp.max(arr, axis=axis, keepdims=True)
    return xp.log(xp.sum(xp.exp(arr - m), axis=axis)) + xp.squeeze(m, axis=axis)


def forward(obs, pi, A, B):
    """Log-space HMM forward algorithm.

    Parameters
    ----------
    obs : 1D int array, length T
        Observation indices in [0, M).
    pi : 1D array, length N
        Initial state distribution.
    A : (N, N) array
        Transition probabilities. Row i sums to 1.
    B : (N, M) array
        Emission probabilities. Row i sums to 1.

    Returns
    -------
    log_alpha : (T, N) array
        Forward log-probabilities. log_alpha[t, i] = log P(o_{0..t}, s_t = i).
    log_likelihood : float
        log P(O | model). Sum-log-exp of the last row of log_alpha.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([0, 1, 0])
    >>> pi = np.array([0.5, 0.5])
    >>> A = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> B = np.array([[0.9, 0.1], [0.2, 0.8]])
    >>> log_alpha, log_lik = forward(obs, pi, A, B)
    >>> log_alpha.shape
    (3, 2)
    """
    xp, obs_arr, log_pi, log_A, log_B = _prepare_hmm_inputs(obs, pi, A, B)
    T = obs_arr.size
    N = log_pi.size

    log_alpha = xp.full((T, N), -xp.inf, dtype=np.float64)

    # t = 0: log_alpha[0, i] = log_pi[i] + log_B[i, o_0]
    log_alpha[0] = log_pi + log_B[:, obs_arr[0]]

    # t = 1..T-1
    for t in range(1, T):
        # For each state j, compute logsumexp_i(log_alpha[t-1, i] + log_A[i, j])
        # Vectorize: broadcast log_alpha[t-1] over j via log_A
        # log_alpha[t-1, :, None] + log_A → (N, N), then logsumexp over axis 0
        combined = log_alpha[t - 1, :, None] + log_A  # (N_prev, N_next)
        log_alpha[t] = _logsumexp_axis(xp, combined, axis=0) + log_B[:, obs_arr[t]]

    log_likelihood = float(_logsumexp_axis(xp, log_alpha[-1], axis=0))
    return log_alpha, log_likelihood

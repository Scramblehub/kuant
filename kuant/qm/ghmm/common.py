"""Shared setup for Gaussian HMM inference (continuous scalar observations).

    obs   — 1D float array, length T
    pi    — 1D array of prior probabilities, shape (N,)
    A     — transition matrix, shape (N, N), rows sum to 1
    mu    — per-state emission means, shape (N,)
    sigma — per-state emission std devs, shape (N,), > 0

Emission likelihood per state:
    B[t, i] = φ((obs[t] - mu[i]) / sigma[i]) / sigma[i]

which is the normal PDF `N(mu[i], sigma[i]²)` evaluated at obs[t].
Computed in log-space directly to avoid ever exponentiating tiny values.
"""

from __future__ import annotations


import numpy as np

from kuant._validation import require_1d, require_expected_shape
from kuant.errors import KuantValueError

from ..hmm.forward import _logsumexp_axis  # reuse

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _prepare_ghmm_inputs(obs, pi, A, mu, sigma):
    """Validate + coerce inputs; precompute log_pi, log_A, log_B.

    log_B[t, i] = log N(obs[t]; mu[i], sigma[i]²)
                = -0.5·log(2π) - log(sigma[i]) - 0.5·((obs[t] - mu[i])/sigma[i])²
    """
    obs_arr = np.asarray(obs, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    A_arr = np.asarray(A, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    require_1d(obs_arr, "obs", kernel="ghmm")
    require_1d(pi_arr, "pi", kernel="ghmm")
    N = pi_arr.size
    require_expected_shape(A_arr, "A", (N, N), kernel="ghmm")
    require_expected_shape(mu_arr, "mu", (N,), kernel="ghmm")
    require_expected_shape(sigma_arr, "sigma", (N,), kernel="ghmm")
    if np.any(sigma_arr <= 0):
        bad_idx = int(np.argmin(sigma_arr))
        raise KuantValueError(
            f"kuant.ghmm: 'sigma' must be positive per state; state "
            f"{bad_idx} has sigma={float(sigma_arr[bad_idx])}.  "
            f"[KE-VAL-POSITIVE]\n"
            f"  → Fix: Gaussian emission requires a positive std per state. "
            f"If a state's variance is estimated ~0, set a small floor "
            f"(e.g. 1e-6) or drop that state"
        )

    with np.errstate(divide="ignore"):
        log_pi = np.log(pi_arr)
        log_A = np.log(A_arr)

    # log_B: (T, N) — broadcast obs (T,1) against (mu, sigma) per state.
    z = (obs_arr[:, None] - mu_arr[None, :]) / sigma_arr[None, :]
    log_B = -_LOG_SQRT_2PI - np.log(sigma_arr)[None, :] - 0.5 * z * z

    return np, obs_arr, log_pi, log_A, log_B


__all__ = ["_prepare_ghmm_inputs", "_logsumexp_axis"]

"""HMM state posteriors (γ) and joint posteriors (ξ).

    γ[t, i]     = P(s_t = i | O, model)                = α[t, i] · β[t, i] / P(O)
    ξ[t, i, j]  = P(s_t = i, s_{t+1} = j | O, model)
                = α[t, i] · A[i, j] · B[j, o_{t+1}] · β[t+1, j] / P(O)

Log-space throughout. Uses forward and backward internally.

Design: docs/kernels/hmm_posterior.md.
"""

from __future__ import annotations


from .backward import backward
from .forward import _prepare_hmm_inputs, forward


def posterior(obs, pi, A, B):
    """State posteriors γ and joint state posteriors ξ.

    Returns
    -------
    gamma : (T, N) array
        γ[t, i] = P(s_t = i | O, model). Rows sum to 1.
    xi : (T-1, N, N) array
        ξ[t, i, j] = P(s_t = i, s_{t+1} = j | O, model). Sums to 1 per t.
    log_likelihood : float
        log P(O | model). Same as the value returned by forward().
    """
    xp, obs_arr, log_pi, log_A, log_B = _prepare_hmm_inputs(obs, pi, A, B)

    log_alpha, log_lik = forward(obs, pi, A, B)
    log_beta = backward(obs, pi, A, B)

    # γ[t, i] = exp(log_alpha[t, i] + log_beta[t, i] - log_lik)
    log_gamma = log_alpha + log_beta - log_lik
    gamma = xp.exp(log_gamma)

    # Renormalize each row to sum to 1 (defends against FP drift).
    gamma = gamma / xp.sum(gamma, axis=1, keepdims=True)

    # ξ[t, i, j] for t = 0..T-2
    # log_xi[t, i, j] = log_alpha[t, i] + log_A[i, j] + log_B[j, o_{t+1}] + log_beta[t+1, j] - log_lik
    log_xi = (
        log_alpha[:-1, :, None]  # (T-1, N, 1)
        + log_A[None, :, :]  # (1, N, N)
        + log_B[:, obs_arr[1:]].T[:, None, :]  # (T-1, 1, N)
        + log_beta[1:, None, :]  # (T-1, 1, N)
        - log_lik
    )
    xi = xp.exp(log_xi)
    # Renormalize each t-slice to sum to 1.
    xi = xi / xp.sum(xi, axis=(1, 2), keepdims=True)

    return gamma, xi, log_lik

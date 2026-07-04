"""Baum-Welch EM for Gaussian-emission HMMs (continuous scalar obs).

Same E-step / M-step structure as the discrete case, with the emission
M-step swapped for Gaussian means and standard deviations:

    μ_new[i]  = Σ_t γ[t, i] · obs[t]        / Σ_t γ[t, i]
    σ²_new[i] = Σ_t γ[t, i] · (obs[t] - μ_new[i])²  / Σ_t γ[t, i]

Two numerical guards specific to Gaussian emissions:

1. **σ floor.** Free EM can drive one state's σ to zero when that state
   captures a single observation. The likelihood diverges and the model
   is useless. We clip σ to `min_sigma` (default: population σ · 1e-4)
   after every M-step. Emitted as a `reseeded_states` entry when hit.

2. **State collapse.** Same as the discrete case — if a state's total
   responsibility is zero, we re-seed its μ from a percentile of the
   observation distribution and its σ from the population σ.

Design: docs/kernels/ghmm_baumwelch.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kuant._validation import (
    require_1d,
    require_positive,
    require_probability,
)
from kuant.errors import KuantValueError

from .posterior import posterior


@dataclass
class GHMMBaumWelchResult:
    """Learned Gaussian-HMM parameters + fit metadata.

    Attributes
    ----------
    pi : (N,) array
    A : (N, N) array
    mu, sigma : (N,) arrays
        Per-state emission mean and standard deviation.
    log_likelihood : float
    log_likelihood_history : (n_iter,) array
    n_iter : int
    converged : bool
    reseeded_states : list[int]
    sigma_floor_hits : int
        Count of M-steps in which some state's σ was clipped to the
        floor. Non-zero means the fit was hitting the degenerate
        regime; consider raising `min_sigma`.
    """

    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    log_likelihood: float
    log_likelihood_history: np.ndarray
    n_iter: int
    converged: bool
    reseeded_states: list[int] = field(default_factory=list)
    sigma_floor_hits: int = 0

    def summary(self) -> str:
        conv = "converged" if self.converged else f"HIT max_iter={self.n_iter}"
        return (
            f"BaumWelch (Gaussian HMM): {conv} in {self.n_iter} iterations\n"
            f"  final log-likelihood:   {self.log_likelihood:+.4f}\n"
            f"  states re-seeded:       {len(self.reseeded_states)}\n"
            f"  σ-floor hits:           {self.sigma_floor_hits}"
        )


def _random_stochastic(shape, rng):
    arr = rng.uniform(0.9, 1.1, size=shape)
    return arr / arr.sum(axis=-1, keepdims=True)


def _reseed_row(arr: np.ndarray, i: int, rng: np.random.Generator) -> None:
    """Re-init a stochastic row that collapsed to zero responsibility."""
    n = arr.shape[-1]
    row = rng.uniform(0.9, 1.1, size=n)
    arr[i] = row / row.sum()


def _init_from_percentiles(obs: np.ndarray, N: int, rng: np.random.Generator):
    """Seed per-state (μ, σ) from evenly-spaced obs percentiles.

    This is a good default: for N=2 you get (25th pct, 75th pct); for
    N=3 you get (17th, 50th, 83rd). Each state's σ starts as the
    population σ / N — small enough for states to separate, large
    enough to have coverage.
    """
    obs_finite = obs[np.isfinite(obs)]
    percentiles = np.linspace(100 / (N + 1), 100 - 100 / (N + 1), N)
    mu = np.percentile(obs_finite, percentiles)
    # Small jitter so identical percentiles (rare) diverge.
    mu = mu + rng.normal(0, np.std(obs_finite) * 1e-3, size=N)
    sigma = np.full(N, float(np.std(obs_finite)) / max(N, 1))
    return mu, sigma


def baumwelch(
    obs,
    n_states: int | None = None,
    pi_init=None,
    A_init=None,
    mu_init=None,
    sigma_init=None,
    max_iter: int = 100,
    tol: float = 1e-4,
    min_sigma: float | None = None,
    seed: int = 0,
) -> GHMMBaumWelchResult:
    """EM training for a Gaussian-emission HMM.

    Parameters
    ----------
    obs : 1D float array
        Continuous scalar observations. Length T. NaN entries are
        rejected — pre-process with `dropna` or interpolation.
    n_states : int, optional
        Number of hidden states. Required if the four init arrays are
        not all supplied.
    pi_init, A_init, mu_init, sigma_init : arrays, optional
        Initial parameter guesses. Pass all four for a warm start; pass
        none to auto-init from `n_states` (state means seeded from
        evenly-spaced observation percentiles).
    max_iter : int, default 100
        Cap on EM iterations.
    tol : float, default 1e-4
        Absolute log-likelihood improvement threshold for convergence.
    min_sigma : float, optional
        Floor on per-state σ, applied after every M-step. If None,
        defaults to `std(obs) * 1e-4`. Prevents variance-collapse
        pathologies where a single point becomes a "state" with
        σ → 0.
    seed : int, default 0
        RNG seed for auto-init and state re-seeding.

    Returns
    -------
    GHMMBaumWelchResult

    Notes
    -----
    - Gaussian HMM likelihood is unbounded above without a σ floor —
      one state can shrink σ to zero on a single observation, driving
      logL to +∞. The `min_sigma` floor makes the objective bounded.
      A non-zero `sigma_floor_hits` in the result is a signal that
      you're near this regime — consider raising `n_states`, adding
      more data, or lifting `min_sigma`.
    - The kernel does NOT choose N for you. Fit multiple N and compare
      via BIC = -2·logL + k·log(T), where k = N + N·(N-1) + 2·N
      (transition rows + Gaussian params) is the free parameter count.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Two-regime series: quiet then stressed.
    >>> T = 800
    >>> quiet = rng.normal(0.001, 0.008, T // 2)
    >>> stressed = rng.normal(-0.002, 0.025, T // 2)
    >>> obs = np.concatenate([quiet, stressed])
    >>> result = baumwelch(obs, n_states=2, seed=0, max_iter=50)
    >>> result.converged
    True
    """
    obs_arr = np.asarray(obs, dtype=np.float64)
    require_1d(obs_arr, "obs", kernel="ghmm.baumwelch")

    if obs_arr.size < 2:
        raise KuantValueError(
            "kuant.ghmm.baumwelch: 'obs' must have length >= 2 to compute "
            f"transitions, got {obs_arr.size}.  [KE-VAL-RANGE]\n"
            f"  → Fix: provide a longer observation sequence"
        )
    if not np.all(np.isfinite(obs_arr)):
        n_bad = int((~np.isfinite(obs_arr)).sum())
        raise KuantValueError(
            f"kuant.ghmm.baumwelch: 'obs' contains {n_bad} non-finite "
            f"values.  [KE-VAL-FINITE]\n"
            f"  → Fix: drop or interpolate NaN/inf before calling — "
            f"e.g. `obs = obs[np.isfinite(obs)]`"
        )

    require_positive(max_iter, "max_iter", kernel="ghmm.baumwelch", kind="int")
    require_probability(tol, "tol", kernel="ghmm.baumwelch")

    have_init = all(x is not None for x in (pi_init, A_init, mu_init, sigma_init))
    if not have_init:
        if n_states is None:
            raise KuantValueError(
                "kuant.ghmm.baumwelch: must supply either all of "
                "(pi_init, A_init, mu_init, sigma_init) OR n_states.  "
                "[KE-VAL-MUTEX]\n"
                "  → Fix: `baumwelch(obs, n_states=2)` for auto init, "
                "or pass explicit initial arrays"
            )
        require_positive(n_states, "n_states", kernel="ghmm.baumwelch", kind="int")

    rng = np.random.default_rng(seed)

    if have_init:
        pi = np.asarray(pi_init, dtype=np.float64).copy()
        A = np.asarray(A_init, dtype=np.float64).copy()
        mu = np.asarray(mu_init, dtype=np.float64).copy()
        sigma = np.asarray(sigma_init, dtype=np.float64).copy()
        N = pi.size
    else:
        N = int(n_states)
        pi = _random_stochastic((N,), rng)
        A = _random_stochastic((N, N), rng)
        mu, sigma = _init_from_percentiles(obs_arr, N, rng)

    if min_sigma is None:
        obs_std = float(np.std(obs_arr))
        min_sigma = max(obs_std * 1e-4, 1e-12)
    if not np.all(sigma > 0):
        # Init-time guard — user supplied a zero σ.
        bad = int(np.argmin(sigma))
        raise KuantValueError(
            f"kuant.ghmm.baumwelch: initial 'sigma[{bad}]' must be > 0, "
            f"got {float(sigma[bad])}.  [KE-VAL-POSITIVE]\n"
            f"  → Fix: initialize each state's σ above the min_sigma "
            f"floor ({min_sigma:.3e})"
        )

    log_lik_history: list[float] = []
    reseeded_states: list[int] = []
    sigma_floor_hits = 0
    prev_logL = -np.inf
    converged = False

    for it in range(int(max_iter)):
        # E-step.
        gamma, xi, log_lik = posterior(obs_arr, pi, A, mu, sigma)
        log_lik_history.append(float(log_lik))

        # Monotonicity check with FP slack.
        if it >= 1:
            slack = max(abs(prev_logL) * 1e-9, 1e-10)
            if log_lik + slack < prev_logL:
                raise KuantValueError(
                    f"kuant.ghmm.baumwelch: log-likelihood decreased from "
                    f"{prev_logL:+.6f} to {log_lik:+.6f} at iteration "
                    f"{it}, violating Baum's inequality by "
                    f"{prev_logL - log_lik:.2e}.  [KE-CONV-MONOTONE]\n"
                    f"  → This indicates a numerical or logic bug — "
                    f"raise an issue with the observation sequence and "
                    f"init used"
                )

        if it >= 1 and (log_lik - prev_logL) < tol:
            converged = True
            break
        prev_logL = log_lik

        # M-step.
        pi = gamma[0].copy()

        gamma_denom_A = gamma[:-1].sum(axis=0)  # (N,)
        gamma_denom_full = gamma.sum(axis=0)  # (N,)
        xi_num = xi.sum(axis=0)  # (N, N)

        with np.errstate(divide="ignore", invalid="ignore"):
            A_new = xi_num / gamma_denom_A[:, None]
            mu_new = (gamma * obs_arr[:, None]).sum(axis=0) / gamma_denom_full
            # σ²[i] = Σ_t γ[t,i] · (obs[t] - μ_new[i])²  / Σ_t γ[t,i]
            resid = obs_arr[:, None] - mu_new[None, :]
            var_new = (gamma * resid * resid).sum(axis=0) / gamma_denom_full
            sigma_new = np.sqrt(var_new)

        # State-collapse defense: re-seed states with zero total resp.
        for i in range(N):
            if not np.all(np.isfinite(A_new[i])) or gamma_denom_A[i] == 0:
                _reseed_row(A_new, i, rng)
                reseeded_states.append(i)
            if (
                not np.isfinite(mu_new[i])
                or not np.isfinite(sigma_new[i])
                or gamma_denom_full[i] == 0
            ):
                # Re-seed from observation percentiles.
                mu_seed, sigma_seed = _init_from_percentiles(obs_arr, N, rng)
                mu_new[i] = mu_seed[i]
                sigma_new[i] = sigma_seed[i]
                if i not in reseeded_states:
                    reseeded_states.append(i)

        # σ floor — always applied after M-step to bound the likelihood.
        floor_mask = sigma_new < min_sigma
        if floor_mask.any():
            sigma_new = np.where(floor_mask, min_sigma, sigma_new)
            sigma_floor_hits += 1

        # Exact row-sum re-normalization on A.
        A_new = A_new / A_new.sum(axis=1, keepdims=True)

        A = A_new
        mu = mu_new
        sigma = sigma_new

    return GHMMBaumWelchResult(
        pi=pi,
        A=A,
        mu=mu,
        sigma=sigma,
        log_likelihood=float(log_lik_history[-1]),
        log_likelihood_history=np.asarray(log_lik_history),
        n_iter=len(log_lik_history),
        converged=converged,
        reseeded_states=reseeded_states,
        sigma_floor_hits=sigma_floor_hits,
    )


__all__ = ["baumwelch", "GHMMBaumWelchResult"]

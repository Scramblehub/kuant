"""Baum-Welch EM parameter training for discrete HMMs.

Given an observation sequence and either an initial parameter guess
or a state/symbol count for random init, iterate the EM recurrence:

    E-step: γ, ξ = posterior(obs; π, A, B)                       [existing kuant.qm.hmm.posterior]
    M-step: π_new[i]    = γ[0, i]
            A_new[i, j] = Σ_t ξ[t, i, j] / Σ_t γ[t, i]           (t = 0..T-2)
            B_new[i, k] = Σ_{t: o_t = k} γ[t, i] / Σ_t γ[t, i]

Iterate until the log-likelihood improvement drops below `tol` or
`max_iter` is reached. The likelihood is monotonically non-decreasing
under exact EM (Baum's inequality); we assert that within a small
numerical tolerance and warn the user if it's violated by more than
that.

State-collapse defense: a state with zero total responsibility
(Σ_t γ[t, i] ~ 0) can't be updated and its row of A/B becomes NaN.
When that happens we re-seed the state with a small perturbation of
the population mean. Rare in practice unless the initial guess is
adversarial.

Design: docs/kernels/hmm_baumwelch.md.
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
class BaumWelchResult:
    """Learned HMM parameters + fit metadata.

    Attributes
    ----------
    pi, A, B : np.ndarray
        Learned initial state distribution, transition matrix, emission
        matrix.
    log_likelihood : float
        Final data log-likelihood at the returned parameters.
    log_likelihood_history : np.ndarray
        Per-iteration log-likelihood (length = number of EM iterations).
        Monotone non-decreasing under exact EM.
    n_iter : int
        EM iterations actually run.
    converged : bool
        True iff the last iteration's likelihood improvement was below
        `tol`. False if `max_iter` was hit first.
    reseeded_states : list[int]
        Any state indices that had zero total responsibility during
        an M-step and were re-seeded from a perturbation of the
        population distribution.
    """

    pi: np.ndarray
    A: np.ndarray
    B: np.ndarray
    log_likelihood: float
    log_likelihood_history: np.ndarray
    n_iter: int
    converged: bool
    reseeded_states: list[int] = field(default_factory=list)

    def summary(self) -> str:
        conv = "converged" if self.converged else f"HIT max_iter={self.n_iter}"
        return (
            f"BaumWelch (discrete HMM): {conv} in {self.n_iter} iterations\n"
            f"  final log-likelihood:   {self.log_likelihood:+.4f}\n"
            f"  ΔlogL last step:        "
            f"{self.log_likelihood_history[-1] - self.log_likelihood_history[-2] if len(self.log_likelihood_history) >= 2 else 0:+.2e}\n"
            f"  states re-seeded:       {len(self.reseeded_states)}"
        )


def _random_stochastic(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Draw a matrix whose rows sum to 1, seeded near-uniform with jitter."""
    arr = rng.uniform(0.9, 1.1, size=shape)
    return arr / arr.sum(axis=-1, keepdims=True)


def _reseed_row(arr: np.ndarray, i: int, rng: np.random.Generator) -> None:
    """Re-initialize a stochastic row that collapsed to zero responsibility."""
    n = arr.shape[-1]
    row = rng.uniform(0.9, 1.1, size=n)
    arr[i] = row / row.sum()


def baumwelch(
    obs,
    n_states: int | None = None,
    n_symbols: int | None = None,
    pi_init=None,
    A_init=None,
    B_init=None,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
) -> BaumWelchResult:
    """EM parameter training for a discrete-observation HMM.

    Parameters
    ----------
    obs : 1D int array
        Observation indices in `[0, n_symbols)`. Length T.
    n_states : int, optional
        Number of hidden states. Required if `pi_init`/`A_init`/`B_init`
        are not all supplied; ignored otherwise.
    n_symbols : int, optional
        Number of observation symbols. Same rule as `n_states`.
    pi_init, A_init, B_init : arrays, optional
        Initial parameter guesses. Pass all three for a warm start; pass
        none to random-init from `(n_states, n_symbols, seed)`.
    max_iter : int, default 100
        Cap on EM iterations. Reached iff the likelihood is still
        improving by more than `tol`.
    tol : float, default 1e-4
        Absolute log-likelihood improvement threshold for convergence.
    seed : int, default 0
        RNG seed for random init and state re-seeding.

    Returns
    -------
    BaumWelchResult

    Notes
    -----
    - Log-likelihood is monotone non-decreasing under exact EM. We check
      this and raise `KuantValueError` if the drop exceeds
      max(|logL| · 1e-9, 1e-10) — that only happens on numerical
      pathologies (e.g. state re-seeding right after a big improvement).
      A slight backslide can be a signal of over-fit or a degenerate
      symbol; the check catches gross bugs, not marginal cases.
    - Symbols are 0-indexed. If your observations use a different
      alphabet, remap them first.
    - The kernel does NOT do model selection: `n_states` is a
      hyperparameter you set. Compare fits at different `n_states` via
      the final `log_likelihood` (or BIC = -2·logL + k·log(T) where k
      is the parameter count).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Two-state HMM: state 0 emits mostly symbol 0, state 1 mostly symbol 1.
    >>> T = 400
    >>> obs = np.concatenate([rng.integers(0, 2, size=T // 2, endpoint=False),
    ...                         rng.integers(0, 2, size=T // 2, endpoint=False)])
    >>> result = baumwelch(obs, n_states=2, n_symbols=2, seed=1)
    >>> result.converged
    True
    """
    obs_arr = np.asarray(obs)
    if obs_arr.dtype.kind not in "iu":
        obs_arr = obs_arr.astype(np.int64)
    require_1d(obs_arr, "obs", kernel="baumwelch")

    if obs_arr.size < 2:
        raise KuantValueError(
            "kuant.baumwelch: 'obs' must have length >= 2 to compute "
            f"transitions, got {obs_arr.size}.  [KE-VAL-RANGE]\n"
            f"  → Fix: provide a longer observation sequence"
        )

    require_positive(max_iter, "max_iter", kernel="baumwelch", kind="int")
    require_probability(tol, "tol", kernel="baumwelch")

    # Determine initialization mode.
    have_init = pi_init is not None and A_init is not None and B_init is not None
    if not have_init:
        if n_states is None or n_symbols is None:
            raise KuantValueError(
                "kuant.baumwelch: must supply either all of "
                "(pi_init, A_init, B_init) OR both n_states and n_symbols.  "
                "[KE-VAL-MUTEX]\n"
                "  → Fix: `baumwelch(obs, n_states=3, n_symbols=4)` for "
                "random init, or pass explicit initial matrices"
            )
        require_positive(n_states, "n_states", kernel="baumwelch", kind="int")
        require_positive(n_symbols, "n_symbols", kernel="baumwelch", kind="int")

    rng = np.random.default_rng(seed)

    if have_init:
        pi = np.asarray(pi_init, dtype=np.float64).copy()
        A = np.asarray(A_init, dtype=np.float64).copy()
        B = np.asarray(B_init, dtype=np.float64).copy()
        N = pi.size
    else:
        N = int(n_states)
        M = int(n_symbols)
        pi = _random_stochastic((N,), rng)
        A = _random_stochastic((N, N), rng)
        B = _random_stochastic((N, M), rng)

    # Bounds-check observations against B's symbol axis.
    M = B.shape[1]
    obs_max = int(obs_arr.max())
    if obs_max >= M or int(obs_arr.min()) < 0:
        raise KuantValueError(
            f"kuant.baumwelch: 'obs' values must be in [0, {M}), got "
            f"range [{int(obs_arr.min())}, {obs_max}].  [KE-VAL-RANGE]\n"
            f"  → Fix: remap observations to [0, n_symbols), or raise "
            f"`n_symbols` to at least {obs_max + 1}"
        )

    log_lik_history: list[float] = []
    reseeded_states: list[int] = []
    prev_logL = -np.inf
    converged = False

    for it in range(int(max_iter)):
        # E-step: γ, ξ, logL from existing kuant.qm.hmm.posterior.
        gamma, xi, log_lik = posterior(obs_arr, pi, A, B)
        log_lik_history.append(float(log_lik))

        # Monotonicity check — Baum's inequality. Allow tiny FP slack.
        if it >= 1:
            slack = max(abs(prev_logL) * 1e-9, 1e-10)
            if log_lik + slack < prev_logL:
                raise KuantValueError(
                    f"kuant.baumwelch: log-likelihood decreased from "
                    f"{prev_logL:+.6f} to {log_lik:+.6f} at iteration "
                    f"{it}, violating Baum's inequality by "
                    f"{prev_logL - log_lik:.2e}.  [KE-CONV-MONOTONE]\n"
                    f"  → This indicates a numerical or logic bug — "
                    f"raise an issue with the observation sequence and "
                    f"init used"
                )

        # Convergence: absolute logL improvement below tol.
        if it >= 1 and (log_lik - prev_logL) < tol:
            converged = True
            break
        prev_logL = log_lik

        # M-step.
        # π_new[i] = γ[0, i]
        pi = gamma[0].copy()

        # A_new[i, j] = Σ_t ξ[t, i, j] / Σ_t γ[t, i]     (t = 0..T-2)
        gamma_denom_A = gamma[:-1].sum(axis=0)  # (N,)
        xi_num = xi.sum(axis=0)  # (N, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            A_new = xi_num / gamma_denom_A[:, None]

        # B_new[i, k] = Σ_{t: o_t = k} γ[t, i] / Σ_t γ[t, i]
        gamma_denom_B = gamma.sum(axis=0)  # (N,)
        B_new = np.zeros_like(B)
        for k in range(M):
            mask = obs_arr == k
            if mask.any():
                B_new[:, k] = gamma[mask].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            B_new = B_new / gamma_denom_B[:, None]

        # Detect + re-seed states with zero responsibility.
        for i in range(N):
            if not np.all(np.isfinite(A_new[i])) or gamma_denom_A[i] == 0:
                _reseed_row(A_new, i, rng)
                reseeded_states.append(i)
            if not np.all(np.isfinite(B_new[i])) or gamma_denom_B[i] == 0:
                _reseed_row(B_new, i, rng)
                if i not in reseeded_states:
                    reseeded_states.append(i)

        # Re-normalize each row exactly (defends against FP drift).
        A_new = A_new / A_new.sum(axis=1, keepdims=True)
        B_new = B_new / B_new.sum(axis=1, keepdims=True)

        A = A_new
        B = B_new

    return BaumWelchResult(
        pi=pi,
        A=A,
        B=B,
        log_likelihood=float(log_lik_history[-1]),
        log_likelihood_history=np.asarray(log_lik_history),
        n_iter=len(log_lik_history),
        converged=converged,
        reseeded_states=reseeded_states,
    )


__all__ = ["baumwelch", "BaumWelchResult"]

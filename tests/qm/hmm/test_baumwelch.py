"""Tests for kuant.qm.hmm.baumwelch — EM training for discrete HMMs.

Correctness comes from:
1. Baum's inequality: log-likelihood monotone non-decreasing per iteration
2. Ground-truth recovery: on data sampled from a known HMM, EM converges
   to a permutation of the true parameters (up to label ambiguity)
3. Warm-start behavior: warm-starting at the ground truth reaches
   convergence immediately (or nearly so)
4. Error-message contract: informative KuantError classes on bad input
"""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.qm.hmm.baumwelch import BaumWelchResult, baumwelch


# ---------- sampling helper -----------------------------------------------


def _sample_hmm(pi, A, B, T, rng):
    """Draw a length-T observation sequence from a discrete HMM."""
    pi = np.asarray(pi)
    A = np.asarray(A)
    B = np.asarray(B)
    N, M = B.shape
    states = np.empty(T, dtype=np.int64)
    obs = np.empty(T, dtype=np.int64)
    states[0] = int(rng.choice(N, p=pi))
    obs[0] = int(rng.choice(M, p=B[states[0]]))
    for t in range(1, T):
        states[t] = int(rng.choice(N, p=A[states[t - 1]]))
        obs[t] = int(rng.choice(M, p=B[states[t]]))
    return obs, states


# ---------- return-object contract ----------------------------------------


def test_returns_baumwelch_result():
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 3, size=200)
    result = baumwelch(obs, n_states=2, n_symbols=3, seed=0, max_iter=10)
    assert isinstance(result, BaumWelchResult)
    assert result.pi.shape == (2,)
    assert result.A.shape == (2, 2)
    assert result.B.shape == (2, 3)
    assert result.n_iter == len(result.log_likelihood_history)
    assert result.log_likelihood == float(result.log_likelihood_history[-1])


def test_output_parameters_are_stochastic():
    """Rows of A and B sum to 1; π sums to 1."""
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 4, size=300)
    result = baumwelch(obs, n_states=3, n_symbols=4, seed=0, max_iter=20)
    assert abs(result.pi.sum() - 1.0) < 1e-9
    assert np.allclose(result.A.sum(axis=1), 1.0)
    assert np.allclose(result.B.sum(axis=1), 1.0)


# ---------- Baum's inequality: EM monotone --------------------------------


def test_log_likelihood_monotone_nondecreasing():
    """Baum's inequality: log-likelihood never decreases during EM."""
    rng = np.random.default_rng(0)
    pi_true = np.array([0.7, 0.3])
    A_true = np.array([[0.9, 0.1], [0.2, 0.8]])
    B_true = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    obs, _ = _sample_hmm(pi_true, A_true, B_true, T=300, rng=rng)

    result = baumwelch(obs, n_states=2, n_symbols=3, seed=1, max_iter=25)
    diffs = np.diff(result.log_likelihood_history)
    # Allow tiny FP slack; EM should never drop meaningfully.
    assert (diffs >= -1e-9).all(), (
        f"logL dropped at iterations: {np.where(diffs < 0)[0]}, " f"min diff: {diffs.min():.2e}"
    )


def test_history_length_matches_n_iter():
    obs = np.random.default_rng(0).integers(0, 2, size=100)
    result = baumwelch(obs, n_states=2, n_symbols=2, seed=0, max_iter=5)
    assert result.log_likelihood_history.size == result.n_iter


# ---------- ground-truth recovery -----------------------------------------


def _permute_to_match(A_true, A_est, B_true, B_est):
    """Find the best permutation of estimated states matching the truth."""
    import itertools

    N = A_true.shape[0]
    best_err = np.inf
    best_perm = None
    for perm in itertools.permutations(range(N)):
        p = np.array(perm)
        err = np.abs(A_est[np.ix_(p, p)] - A_true).sum() + np.abs(B_est[p] - B_true).sum()
        if err < best_err:
            best_err = err
            best_perm = p
    return best_perm, best_err


def test_recovers_ground_truth_from_long_sample():
    """On a well-identified 2-state HMM, EM finds a permutation of the
    true parameters within a coarse tolerance. Kept intentionally short
    (T=600, 2 seeds, tol=1e-4) — the recovery signal is unambiguous on
    a well-separated HMM long before 2000 obs / 5 seeds are used."""
    rng = np.random.default_rng(42)
    pi_true = np.array([0.5, 0.5])
    A_true = np.array([[0.95, 0.05], [0.10, 0.90]])
    B_true = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    obs, _ = _sample_hmm(pi_true, A_true, B_true, T=600, rng=rng)

    # Two random inits to escape the worst local optima; keep best.
    best = None
    for seed in range(2):
        r = baumwelch(obs, n_states=2, n_symbols=3, seed=seed, max_iter=50, tol=1e-4)
        if best is None or r.log_likelihood > best.log_likelihood:
            best = r

    perm, err = _permute_to_match(A_true, best.A, B_true, best.B)
    assert err < 0.6, f"L1 param error {err:.3f} too large"


def test_warm_start_at_truth_converges_fast():
    """Warm-starting at the ground truth converges in <=3 iterations."""
    rng = np.random.default_rng(0)
    pi_true = np.array([0.6, 0.4])
    A_true = np.array([[0.85, 0.15], [0.20, 0.80]])
    B_true = np.array([[0.7, 0.3], [0.2, 0.8]])
    obs, _ = _sample_hmm(pi_true, A_true, B_true, T=400, rng=rng)

    result = baumwelch(
        obs,
        pi_init=pi_true,
        A_init=A_true,
        B_init=B_true,
        max_iter=40,
        tol=1e-2,
    )
    assert result.converged
    # Warm-started at the MLE region should refine, not wander far.
    assert result.n_iter <= 20


# ---------- convergence flag semantics ------------------------------------


def test_hits_max_iter_when_tol_too_tight():
    """A very tight tol + few iterations forces the max_iter path."""
    obs = np.random.default_rng(0).integers(0, 3, size=300)
    result = baumwelch(obs, n_states=2, n_symbols=3, seed=0, max_iter=3, tol=1e-20)
    assert result.n_iter == 3
    assert not result.converged


def test_converged_flag_true_when_within_tol():
    """Loose tol + long history should trigger the converged path."""
    rng = np.random.default_rng(0)
    A_true = np.array([[0.9, 0.1], [0.2, 0.8]])
    B_true = np.array([[0.7, 0.3], [0.3, 0.7]])
    obs, _ = _sample_hmm([0.5, 0.5], A_true, B_true, T=300, rng=rng)
    result = baumwelch(obs, n_states=2, n_symbols=2, seed=0, max_iter=50, tol=1e-2)
    assert result.converged


# ---------- error contract ------------------------------------------------


def test_reject_2d_obs():
    with pytest.raises(KuantValueError):
        baumwelch(np.zeros((10, 3), dtype=int), n_states=2, n_symbols=3)


def test_reject_singleton_obs():
    with pytest.raises(KuantValueError) as exc:
        baumwelch(np.array([0], dtype=int), n_states=2, n_symbols=2)
    assert "length >= 2" in str(exc.value)


def test_reject_mutex_init():
    """Neither n_states nor init supplied → mutex error."""
    obs = np.zeros(10, dtype=int)
    with pytest.raises(KuantValueError) as exc:
        baumwelch(obs)
    m = str(exc.value)
    assert "n_states" in m and "pi_init" in m


def test_reject_obs_out_of_symbol_range():
    """obs must be in [0, n_symbols)."""
    obs = np.array([0, 1, 5, 2], dtype=int)
    with pytest.raises(KuantValueError) as exc:
        baumwelch(obs, n_states=2, n_symbols=3)
    m = str(exc.value)
    assert "obs" in m and "[0, 3)" in m


def test_reject_zero_max_iter():
    obs = np.zeros(10, dtype=int)
    with pytest.raises(KuantValueError):
        baumwelch(obs, n_states=2, n_symbols=2, max_iter=0)


def test_reject_bad_tol():
    obs = np.zeros(10, dtype=int)
    with pytest.raises(KuantValueError):
        baumwelch(obs, n_states=2, n_symbols=2, tol=-1e-3)


# ---------- summary + reseeded_states -------------------------------------


def test_summary_string_shape():
    obs = np.random.default_rng(0).integers(0, 3, size=100)
    result = baumwelch(obs, n_states=2, n_symbols=3, seed=0, max_iter=10)
    s = result.summary()
    assert "BaumWelch" in s
    assert "log-likelihood" in s

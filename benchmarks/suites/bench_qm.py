"""Benchmarks for kuant.qm — HMM/GHMM inference, regime discovery tools."""

from __future__ import annotations

import numpy as np

from kuant.qm.ghmm import backward as ghmm_backward
from kuant.qm.ghmm import forward as ghmm_forward
from kuant.qm.ghmm import posterior as ghmm_posterior
from kuant.qm.ghmm import viterbi as ghmm_viterbi
from kuant.qm.hmm import backward as hmm_backward
from kuant.qm.hmm import forward as hmm_forward
from kuant.qm.hmm import posterior as hmm_posterior
from kuant.qm.hmm import viterbi as hmm_viterbi


def _discrete_setup(rng, n=2000, K=3):
    """Discrete HMM: 3 hidden states, 4 observation symbols."""
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.10, 0.85, 0.05],
            [0.05, 0.10, 0.85],
        ]
    )
    B = np.array(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    obs = rng.integers(0, 4, size=n)
    return obs, pi, A, B


def _continuous_setup(rng, n=2000):
    """Two-state Gaussian HMM: quiet + stress."""
    pi = np.array([0.9, 0.1])
    A = np.array([[0.98, 0.02], [0.10, 0.90]])
    mu = np.array([0.0005, -0.001])
    sigma = np.array([0.008, 0.025])
    obs = rng.normal(0, 0.012, size=n)
    return obs, pi, A, mu, sigma


# ---------- Discrete HMM ---------------------------------------------------


def test_bench_hmm_forward_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, B = _discrete_setup(rng, n=2000)
    benchmark(hmm_forward, obs, pi, A, B)


def test_bench_hmm_backward_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, B = _discrete_setup(rng, n=2000)
    benchmark(hmm_backward, obs, pi, A, B)


def test_bench_hmm_viterbi_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, B = _discrete_setup(rng, n=2000)
    benchmark(hmm_viterbi, obs, pi, A, B)


def test_bench_hmm_posterior_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, B = _discrete_setup(rng, n=2000)
    benchmark(hmm_posterior, obs, pi, A, B)


# ---------- Continuous / Gaussian-emission HMM ------------------------------


def test_bench_ghmm_forward_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, mu, sigma = _continuous_setup(rng, n=2000)
    benchmark(ghmm_forward, obs, pi, A, mu, sigma)


def test_bench_ghmm_backward_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, mu, sigma = _continuous_setup(rng, n=2000)
    benchmark(ghmm_backward, obs, pi, A, mu, sigma)


def test_bench_ghmm_viterbi_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, mu, sigma = _continuous_setup(rng, n=2000)
    benchmark(ghmm_viterbi, obs, pi, A, mu, sigma)


def test_bench_ghmm_posterior_2k(benchmark, rng=np.random.default_rng(0)):
    obs, pi, A, mu, sigma = _continuous_setup(rng, n=2000)
    benchmark(ghmm_posterior, obs, pi, A, mu, sigma)
